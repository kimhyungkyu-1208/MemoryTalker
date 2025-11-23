"""
Paper: MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization (ICCV 2025 Submission)
File: models/memory_talker_stage2.py

Description: 
Defines MemoryTalker architecture for Stage 2.
Includes SpeakingStyleEncoder and Stylized Motion Memory logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from HuBERT import HubertForCTC
from models.audio_encoder_for_style import SpeakerEncoder, extract_logmel_torchaudio_tensor

# Reuse Stage 1 components where possible (or redefine for standalone file)
# Here we redefine for completeness based on provided code

def linear_interpolation(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None: output_len = int(seq_len * output_fps)
    return F.interpolate(features, size=output_len, align_corners=True, mode='linear').transpose(1, 2)

def compute_velocity_loss(gt, estimated):
    gt_vel = gt[:, 1:, :] - gt[:, :-1, :]
    est_vel = estimated[:, 1:, :] - estimated[:, :-1, :]
    return F.mse_loss(gt_vel, est_vel)

class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=30, max_seq_len=600):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).repeat(1, max_seq_len // period + 1, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class FacialMotionMemory(nn.Module):
    """
    Handles both General Motion (Stage 1) and Stylized Motion (Stage 2).
    """
    def __init__(self, args):
        super().__init__()
        self.feature_dim = args.feature_dim
        self.mem_slot = args.mem_slot
        self.scaling_term = args.softmax_scaling_term
        
        # M_m: Base Motion Memory
        self.M_m = nn.Parameter(torch.Tensor(self.mem_slot, self.feature_dim))
        nn.init.xavier_normal_(self.M_m)

    def get_feature(self, addressing, memory):
        # Retrieve: (B, T, n) x (n, d) -> (B, T, d)
        return torch.einsum('bts,sd->btd', addressing, memory)

    def forward(self, f_txt, w_s):
        """
        Stage 2 Forward: Stylized Retrieval
        f_txt: Text features (Queries)
        w_s: Style weights from StyleEncoder (B, n) or (n,)
        """
        # Eq 14: Stylized Motion Memory M_m_tilde
        # M_m_tilde = M_m * w_s (Element-wise scaling per slot)
        # Assuming w_s is (n_slots, 1) or broadcastable
        if w_s.dim() == 1:
            w_s = w_s.unsqueeze(-1) # (n, 1)
        
        # Stylize Memory: scale each slot i by weight w_s^i
        # M_m: (n, d), w_s: (n, 1) -> M_m_tilde: (n, d)
        # Note: Original code used einsum 's,sd->sd', implying per-sample style or shared style?
        # Typically style is per-speaker. If w_s is per batch, we need careful broadcasting.
        # Original code: 's,sd->sd' implies w_s is (Slot,) and M_m is (Slot, Dim). 
        # This implies w_s is averaged or single vector. 
        
        M_m_tilde = w_s * self.M_m # Broadcasting
        
        # Eq 5: Key Addressing
        K_txt = F.softmax(self.scaling_term * f_txt, dim=-1) # (B, T, n)
        
        # Retrieve
        f_m_key = self.get_feature(K_txt, M_m_tilde)
        return f_m_key

class SpeakingStyleEncoder(nn.Module):
    """
    Section 3.2: Extracts Style Features (f_s) and computes Style Weights (w_s).
    Also computes Triplet Loss (L_style).
    """
    def __init__(self, args):
        super().__init__()
        self.mem_slot = args.mem_slot
        self.triplet_margin = args.triplet_margin
        self.device = args.device
        
        # Audio Encoder Backbone
        self.audio_encoder = SpeakerEncoder()
        self.audio_encoder.initialize_weights()

        # Projections
        self.to_style_emb = nn.Linear(256, self.mem_slot) # Maps to slot dimension
        self.to_scale = nn.Linear(self.mem_slot, 1, bias=False) # Maps to scalar scaling factor
        
        # MLP for Triplet Metric Learning
        self.style_mlp = nn.Sequential(
            nn.Linear(self.mem_slot, self.mem_slot),
            nn.ReLU(),
            nn.Linear(self.mem_slot, self.mem_slot)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    def forward(self, audio_anchor, audio_neg=None, audio_pos=None):
        # 1. Extract f_s (Style Feature)
        mel_anchor = extract_logmel_torchaudio_tensor(audio_anchor)
        f_s = self.to_style_emb(self.audio_encoder(mel_anchor)) # (B, n_slots)
        
        # 2. Compute Style Weights w_s (Eq 15)
        # w_s = Sigmoid(psi'(f_s)) * psi(f_s)
        # Original code: sigmoid(f_s) * linear(f_s)
        # Here assuming f_s is the vector to be sigmoid-ed
        scale_factor = self.to_scale(f_s) # (B, 1)
        w_s = torch.sigmoid(f_s) * scale_factor # (B, n_slots)

        # 3. Triplet Loss (L_style) - Eq 13
        loss_style = torch.tensor(0.0, device=self.device)
        if audio_neg is not None and audio_pos is not None:
            mel_neg = extract_logmel_torchaudio_tensor(audio_neg)
            mel_pos = extract_logmel_torchaudio_tensor(audio_pos)
            
            f_s_neg = self.to_style_emb(self.audio_encoder(mel_neg))
            f_s_pos = self.to_style_emb(self.audio_encoder(mel_pos))
            
            # Project to metric space
            anchor_proj = self.style_mlp(f_s)
            pos_proj = self.style_mlp(f_s_pos)
            neg_proj = self.style_mlp(f_s_neg)
            
            loss_style = F.triplet_margin_loss(anchor_proj, pos_proj, neg_proj, margin=self.triplet_margin)
            
        return w_s, loss_style

class MemoryTalker(nn.Module):
    """
    MemoryTalker Stage 2: Audio-Guided Stylization.
    Combines ASR features with Stylized Motion Memory.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        
        # Modules
        self.a2t_model = HubertForCTC.from_pretrained("facebook/hubert-base-ls960")
        self.a2t_model.freeze_feature_encoder()
        
        self.txt_proj = nn.Linear(args.mem_slot, args.feature_dim)
        
        # Motion Decoder components
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period=args.period)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4, dim_feedforward=2*args.feature_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        
        self.motion2txt_map = nn.Linear(args.feature_dim*2, args.feature_dim)
        self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)
        
        # Core Components
        self.memory_net = FacialMotionMemory(args)
        self.style_encoder = SpeakingStyleEncoder(args)
        
        self.dropout = nn.Dropout(0.1 if args.dataset == "BIWI" else 0.0)

    def forward(self, audio, template, vertice, one_hot, inference=False, audio_neg=None, audio_pos=None, lip_indices=None):
        template = template.unsqueeze(1)
        
        # 1. Text Features (f_txt)
        if inference:
            txt_logits = self.a2t_model(audio, self.dataset).logits
            frame_num = txt_logits.shape[1] // 2 if self.dataset == 'BIWI' else txt_logits.shape[1]
        else:
            frame_num = vertice.shape[1]
            txt_logits = self.a2t_model(audio, self.dataset, frame_num=frame_num).logits

        f_txt = self.txt_proj(txt_logits)
        
        # 2. Style Features & Weights (w_s)
        # Note: Triplet loss calculated here
        w_s, loss_style = self.style_encoder(audio, audio_neg, audio_pos)
        
        # Squeeze batch dim for style weight if needed (Assuming batch processing, we average or take per-sample)
        # The original code did `w_s.squeeze()` which implies B=1 or careful broadcasting. 
        # Here we assume w_s is (B, Slots). We map this to (B, 1, Slots) for broadcasting in memory if batch > 1.
        w_s = w_s.squeeze() # Adjust based on batch size logic

        # 3. Retrieve Stylized Motion (Eq 6 with Stylized Memory)
        if self.dataset == 'BIWI':
            f_txt_interp = linear_interpolation(f_txt, 50, 25, frame_num)
            logits_interp = linear_interpolation(txt_logits, 50, 25, frame_num)
            
            f_m_key = self.memory_net(logits_interp, w_s)
            
            # Decoder Inputs
            tgt_f = torch.cat([f_txt_interp, f_m_key], dim=-1)
            # Simple masking for BIWI
            mask = torch.zeros((audio.shape[0], frame_num), dtype=torch.bool, device=self.device) 
        else:
            f_m_key = self.memory_net(txt_logits, w_s)
            tgt_f = torch.cat([f_txt, f_m_key], dim=-1)
            mask = torch.zeros((audio.shape[0], frame_num), dtype=torch.bool, device=self.device)

        # 4. Decode
        proj_tgt = self.PPE(self.motion2txt_map(tgt_f))
        decoded = self.transformer_decoder(proj_tgt, f_txt) # Pass mask if strictly needed
        pred_motion = self.vertice_map_r(decoded) + template

        # 5. Losses
        loss_mse = torch.tensor(0.0, device=self.device)
        loss_vel = torch.tensor(0.0, device=self.device)
        loss_lip = torch.tensor(0.0, device=self.device)

        if not inference:
            # L_mse (Eq 9)
            loss_mse = F.mse_loss(pred_motion, vertice.detach())
            
            # L_vel (Eq 10)
            loss_vel = compute_velocity_loss(vertice.detach(), pred_motion)
            
            # L_lip (Lip Vertex Loss)
            # Needs lip_indices passed from trainer
            if lip_indices is not None:
                # Reshape to (B, T, V, 3)
                B, T, _ = pred_motion.shape
                pred_reshaped = pred_motion.reshape(B, T, -1, 3)
                gt_reshaped = vertice.reshape(B, T, -1, 3)
                
                loss_lip = F.mse_loss(
                    pred_reshaped[:, :, lip_indices, :], 
                    gt_reshaped[:, :, lip_indices, :].detach()
                )

        return pred_motion, loss_mse, loss_vel, 0, 0, loss_style, loss_lip