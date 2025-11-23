"""
Paper: MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization (ICCV 2025 Submission)
File: models/memory_talker.py

Description: 
Defines the core MemoryTalker architecture focusing on Stage 1 (Memorizing).
Includes FacialMotionMemory, Encoders, and Decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from HuBERT import HubertForCTC

def linear_interpolation(features, input_fps, output_fps, output_len=None):
    """
    Interpolates features to match the target FPS (e.g., Audio FPS -> Video FPS).
    """
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features, size=output_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)

def compute_velocity_loss(gt, estimated, reduction='mean'):
    """
    Computes Velocity Loss (Eq 10) to reduce jitter.
    L_vel = ||(v_t+1 - v_t) - (v'_t+1 - v'_t)||^2
    """
    gt_velocity = gt[:, 1:, :] - gt[:, :-1, :]
    estimated_velocity = estimated[:, 1:, :] - estimated[:, :-1, :]
    return F.mse_loss(gt_velocity, estimated_velocity, reduction=reduction)

def get_mask(device, dataset, T, S):
    """
    Creates attention mask for the Transformer Decoder.
    """
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        # BIWI often uses double frame rate or specific masking
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask == 1).to(device=device)

class PeriodicPositionalEncoding(nn.Module):
    """
    Periodic Positional Encoding (PPE) to handle temporal sequences.
    """
    def __init__(self, d_model, dropout=0.1, period=30, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len // period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Slice to current sequence length
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class FacialMotionMemory(nn.Module):
    """
    Section 3.1: Facial Motion Memory (M_m)
    Stores general facial motion features and supports Key-Value retrieval.
    """
    def __init__(self, args):
        super(FacialMotionMemory, self).__init__()
        self.feature_dim = args.feature_dim # c
        self.mem_slot = args.mem_slot       # n
        self.scaling_term = args.softmax_scaling_term # kappa

        # M_m: Motion Memory (n x c)
        self.M_m = nn.Parameter(torch.Tensor(self.mem_slot, self.feature_dim), requires_grad=True)
        nn.init.xavier_normal_(self.M_m)

    def get_similarity(self, query, memory):
        """
        Calculates similarity and attention weights (Eq 1).
        """
        norm_q = F.normalize(query, p=2, dim=-1)  # (B, T, c)
        norm_m = F.normalize(memory, p=2, dim=-1) # (n, c)
        
        # Cosine Similarity: d(s_m^i, f_m^t)
        similarity = torch.einsum('btd,sd->bts', norm_q, norm_m)  # (B, T, n)
        
        # Attention Weights (Eq 1)
        attention_weights = F.softmax(self.scaling_term * similarity, dim=-1)
        return attention_weights

    def retrieve(self, weights, memory):
        """
        Retrieves features via weighted sum (Eq 2, Eq 6).
        """
        # (B, T, n) * (n, c) -> (B, T, c)
        retrieved_features = torch.einsum('bts,sd->btd', weights, memory)
        return retrieved_features

    def update_loss(self, f_txt, f_m):
        """
        Calculates Stage 1 Losses: L_mem (Eq 3) and L_align (Eq 7).
        f_txt: Text representation (Query for general motion)
        f_m: Ground Truth Motion feature (Query for memory update)
        """
        # 1. Text Address (Key Address Vector K_txt) - Eq 5
        K_txt = F.softmax(self.scaling_term * f_txt.detach(), dim=-1)

        # 2. Motion Address (Value Address Vector V_m) - Eq 1
        V_m = self.get_similarity(query=f_m.detach(), memory=self.M_m)

        # 3. Alignment Loss (L_align) - Eq 7 (KL Divergence between K_txt and V_m)
        loss_align = F.kl_div(V_m.log(), K_txt, reduction='batchmean')
        
        # 4. Memory Reconstruction Loss (L_mem) - Eq 3
        # Retrieve f_m_val using V_m (Eq 2)
        f_m_val = self.retrieve(weights=V_m, memory=self.M_m)
        loss_mem = F.mse_loss(f_m.detach(), f_m_val)

        return loss_align, loss_mem
    
    def forward(self, f_txt):
        """
        Inference / Decoding path: Retrieve general motion using Text Features.
        """
        # Eq 5: Key Address Vector
        K_txt = F.softmax(self.scaling_term * f_txt, dim=-1) 
        
        # Eq 6: Retrieve f_m_key
        f_m_key = self.retrieve(weights=K_txt, memory=self.M_m.detach()) # Detach memory during inference lookup? Usually yes for stability
        
        return f_m_key

class MemoryTalker(nn.Module):
    """
    Main MemoryTalker Model (Stage 1 Configuration).
    """
    def __init__(self, args):
        super(MemoryTalker, self).__init__()
        self.device = args.device
        self.dataset = args.dataset
        self.vertice_dim = args.vertice_dim
        self.feature_dim = args.feature_dim
        self.mem_slot = args.mem_slot

        # 1. Audio Encoder (E_aud): HuBERT
        # Pre-trained ASR encoder to extract text representations
        # Note: 'facebook/hubert-base-ls960' should be cached or available
        self.E_aud = HubertForCTC.from_pretrained("facebook/hubert-base-ls960")
        self.E_aud.freeze_feature_encoder()
        
        # Projection to Memory Slot dimension (Eq 4 projection psi)
        self.txt_proj = nn.Linear(self.mem_slot, self.feature_dim) # Technically maps logits -> feature space
        nn.init.xavier_uniform_(self.txt_proj.weight)
        
        # 2. Motion Encoder (E_m)
        # Encodes 3D vertex motion into feature space
        self.E_m = nn.Linear(self.vertice_dim, self.feature_dim)
        nn.init.xavier_uniform_(self.E_m.weight)
        
        # Dropout configuration based on dataset
        self.dropout = nn.Dropout(0.0) 
        
        # 3. Motion Decoder (D_m)
        # Transformer-based decoder
        self.PPE = PeriodicPositionalEncoding(self.feature_dim, period=args.period)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.feature_dim, 
            nhead=4, 
            dim_feedforward=2*self.feature_dim, 
            batch_first=True
        )
        self.D_m = nn.TransformerDecoder(decoder_layer, num_layers=1)
        
        # Fusion Layer (Combining Text + Memory features)
        self.fusion_layer = nn.Linear(self.feature_dim * 2, self.feature_dim)
        nn.init.xavier_uniform_(self.fusion_layer.weight)
        
        # Output Projection
        self.output_proj = nn.Linear(self.feature_dim, self.vertice_dim)
        nn.init.constant_(self.output_proj.weight, 0.0)
        nn.init.constant_(self.output_proj.bias, 0.0)
        
        # 4. Memory Module
        self.memory_net = FacialMotionMemory(args)

    def forward(self, audio, template, vertice, one_hot, inference=False):
        """
        Returns:
            pred_motion (Tensor): Synthesized 3D motion
            loss_mse (Tensor): L_mse
            loss_vel (Tensor): L_vel
            loss_mem (Tensor): L_mem (Stage 1)
            loss_align (Tensor): L_align (Stage 1)
            loss_style (Tensor): L_style (Stage 2 - returns 0 here)
            loss_pair (Tensor): Auxiliary loss (returns 0)
        """
        # Init losses
        loss_mse = torch.tensor(0.0, device=self.device)
        loss_vel = torch.tensor(0.0, device=self.device)
        loss_mem = torch.tensor(0.0, device=self.device)
        loss_align = torch.tensor(0.0, device=self.device)
        
        # Prepare Template
        template = template.unsqueeze(1) # (B, 1, V*3)
        
        # 1. Audio Encoding (A -> Text Logits)
        if not inference:
            # Training: Use GT motion length
            frame_num = vertice.shape[1]
            txt_logits = self.E_aud(audio, self.dataset, frame_num=frame_num).logits
            gt_motion = vertice - template
        else:
            # Inference: Use Audio length
            txt_logits = self.E_aud(audio, self.dataset).logits
            frame_num = txt_logits.shape[1]
            if self.dataset == 'BIWI': frame_num //= 2

        # 2. Text Representation (f_txt)
        f_txt = self.txt_proj(txt_logits) # (B, T, c)

        # Handling Dataset FPS differences (Interpolation)
        if self.dataset == 'BIWI':
            f_txt_interp = linear_interpolation(f_txt, 50, 25, frame_num)
            txt_logits_interp = linear_interpolation(txt_logits, 50, 25, frame_num)
            
            # Retrieve Memory (General Motion)
            f_m_key = self.memory_net(f_txt=txt_logits_interp) # (B, T, c)
            
            decoder_mask = get_mask(self.device, self.dataset, txt_logits.shape[1]//2, txt_logits.shape[1])
            fusion_input = torch.cat([f_txt_interp, f_m_key], dim=-1)
        else:
            # VOCASET
            # Retrieve Memory (General Motion) - Eq 6
            f_m_key = self.memory_net(f_txt=txt_logits) # (B, T, c)
            
            decoder_mask = get_mask(self.device, self.dataset, txt_logits.shape[1], txt_logits.shape[1])
            fusion_input = torch.cat([f_txt, f_m_key], dim=-1)
        
        # 3. Decoding (Eq 8)
        # Combine [f_txt; f_m_key]
        proj_fusion = self.PPE(self.fusion_layer(fusion_input))
        
        # Transformer Decode
        # Query: Combined features, Key/Value: Text features
        decoded_feat = self.D_m(proj_fusion, f_txt, memory_mask=decoder_mask)
        
        # Output Projection
        residual_motion = self.output_proj(decoded_feat)
        pred_motion = residual_motion + template # Final Vertex Position

        # 4. Training Losses
        if not inference:
            # Calculate Motion Features from GT (f_m)
            gt_motion_f = self.dropout(self.E_m(gt_motion))
            
            # --- Memory Update & Consistency Check ---
            if self.dataset == 'BIWI':
                loss_align, loss_mem = self.memory_net.update_loss(f_txt=txt_logits_interp, f_m=gt_motion_f)
                # Teacher Forcing path for stability (Optional but kept from original)
                val_tgt_f = torch.cat([f_txt_interp, gt_motion_f], dim=-1)
                val_mask = decoder_mask
            else:
                loss_align, loss_mem = self.memory_net.update_loss(f_txt=txt_logits, f_m=gt_motion_f)
                val_tgt_f = torch.cat([f_txt, gt_motion_f], dim=-1)
                val_mask = decoder_mask
            
            # Re-run decoder with GT-injected features (Teacher Forcing)
            val_proj_tgt = self.PPE(self.fusion_layer(val_tgt_f))
            val_decoded = self.D_m(val_proj_tgt, f_txt, memory_mask=val_mask)
            val_motion = self.output_proj(val_decoded) + template

            # Losses
            # Eq 9: MSE Loss (Combining pure prediction + teacher forced prediction)
            loss_mse = F.mse_loss(pred_motion, vertice.detach()) + F.mse_loss(val_motion, vertice.detach())
            
            # Eq 10: Velocity Loss
            loss_vel = compute_velocity_loss(pred_motion, vertice.detach()) + compute_velocity_loss(val_motion, vertice.detach())

        # Return tuple matching train.py signature
        # (pred, mse, vel, mem, align, style, pair)
        return pred_motion, loss_mse, loss_vel, loss_mem, loss_align, torch.tensor(0.0), torch.tensor(0.0)