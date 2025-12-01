"""
Paper: MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization (ICCV 2025 Submission)
File: models/memory_talker_stage2.py

Description:
Stage 2 (Animating / Stylization) model for MemoryTalker.
- Module / parameter names are aligned with Stage 1 (memory_talker.py)
  so that Stage 1 checkpoints can be loaded seamlessly.
- Adds SpeakingStyleEncoder and stylized motion memory on top.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from HuBERT import HubertForCTC
from models.audio_encoder_for_style import SpeakerEncoder, extract_logmel_torchaudio_tensor


# ---------------------------
# Common utility functions (Stage1와 동일 이름)
# ---------------------------

def linear_interpolation(features, input_fps, output_fps, output_len=None):
    """
    Interpolates features to match the target FPS (e.g., Audio FPS -> Video FPS).
    """
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features, size=output_len,
                                    align_corners=True, mode='linear')
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
    (Stage1와 동일 로직)
    """
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i * 2:i * 2 + 2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask == 1).to(device=device)


# ---------------------------
# Positional Encoding (Stage1 동일)
# ---------------------------

class PeriodicPositionalEncoding(nn.Module):
    """
    Periodic Positional Encoding (PPE) to handle temporal sequences.
    """
    def __init__(self, d_model, dropout=0.1, period=30, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, period, d_model)
        repeat_num = (max_seq_len // period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Slice to current sequence length
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------
# Facial Motion Memory (Stage1 동일 정의)
# ---------------------------

class FacialMotionMemory(nn.Module):
    """
    Section 3.1: Facial Motion Memory (M_m)
    Stores general facial motion features and supports Key-Value retrieval.
    """
    def __init__(self, args):
        super(FacialMotionMemory, self).__init__()
        self.feature_dim = args.feature_dim   # c
        self.mem_slot = args.mem_slot        # n
        self.scaling_term = args.softmax_scaling_term  # kappa

        # M_m: Motion Memory (n x c)
        self.M_m = nn.Parameter(
            torch.Tensor(self.mem_slot, self.feature_dim), requires_grad=True
        )
        nn.init.xavier_normal_(self.M_m)

    def get_similarity(self, query, memory):
        """
        Calculates similarity and attention weights (Eq 1).
        query: (B, T, c), memory: (n, c)
        """
        norm_q = F.normalize(query, p=2, dim=-1)
        norm_m = F.normalize(memory, p=2, dim=-1)

        # Cosine Similarity: d(s_m^i, f_m^t)
        similarity = torch.einsum('btd,sd->bts', norm_q, norm_m)  # (B, T, n)

        # Attention Weights (Eq 1)
        attention_weights = F.softmax(self.scaling_term * similarity, dim=-1)
        return attention_weights

    def retrieve(self, weights, memory):
        """
        Retrieves features via weighted sum (Eq 2, Eq 6).
        weights: (B, T, n), memory: (n, c)
        """
        retrieved_features = torch.einsum('bts,sd->btd', weights, memory)
        return retrieved_features

    def update_loss(self, f_txt, f_m):
        """
        Stage1에서 사용하는 메모리 정렬/재구성 로스.
        Stage2에서는 사용하지 않지만, state_dict 호환성을 위해 남겨둔다.
        """
        # 1. Text Address (Key Address Vector K_txt) - Eq 5
        K_txt = F.softmax(self.scaling_term * f_txt.detach(), dim=-1)

        # 2. Motion Address (Value Address Vector V_m) - Eq 1
        V_m = self.get_similarity(query=f_m.detach(), memory=self.M_m)

        # 3. Alignment Loss (L_align) - Eq 7
        loss_align = F.kl_div(V_m.log(), K_txt, reduction='batchmean')

        # 4. Memory Reconstruction Loss (L_mem) - Eq 3
        f_m_val = self.retrieve(weights=V_m, memory=self.M_m)
        loss_mem = F.mse_loss(f_m.detach(), f_m_val)

        return loss_align, loss_mem

    def forward(self, f_txt):
        """
        Stage1 Inference / Decoding path: Retrieve general motion using Text Features.
        Stage2에서는 직접 사용하지 않고, retrieve를 통해 stylized memory를 쓸 것임.
        이 forward는 state_dict 호환성을 위해 유지.
        """
        K_txt = F.softmax(self.scaling_term * f_txt, dim=-1)
        f_m_key = self.retrieve(weights=K_txt, memory=self.M_m.detach())
        return f_m_key


# ---------------------------
# Speaking Style Encoder (Stage2 전용)
# ---------------------------

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

        # Style embedding -> slot dimension
        self.to_style_emb = nn.Linear(256, self.mem_slot)
        # Global scaling factor
        self.to_scale = nn.Linear(self.mem_slot, 1, bias=False)

        # MLP for Triplet Metric Learning
        self.style_mlp = nn.Sequential(
            nn.Linear(self.mem_slot, self.mem_slot),
            nn.ReLU(),
            nn.Linear(self.mem_slot, self.mem_slot),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, audio_anchor, audio_neg=None, audio_pos=None):
        # 1) Anchor style feature
        mel_anchor = extract_logmel_torchaudio_tensor(audio_anchor)
        f_s = self.to_style_emb(self.audio_encoder(mel_anchor))  # (B, n_slots)

        # 2) Style weights w_s (slot-wise)
        scale_factor = self.to_scale(f_s)  # (B, 1)
        w_s = torch.sigmoid(f_s) * scale_factor  # (B, n_slots)

        # 3) Triplet loss
        loss_style = torch.tensor(0.0, device=self.device)
        if audio_neg is not None and audio_pos is not None:
            mel_neg = extract_logmel_torchaudio_tensor(audio_neg)
            mel_pos = extract_logmel_torchaudio_tensor(audio_pos)

            f_s_neg = self.to_style_emb(self.audio_encoder(mel_neg))
            f_s_pos = self.to_style_emb(self.audio_encoder(mel_pos))

            anchor_proj = self.style_mlp(f_s)
            pos_proj = self.style_mlp(f_s_pos)
            neg_proj = self.style_mlp(f_s_neg)

            loss_style = F.triplet_margin_loss(
                anchor_proj, pos_proj, neg_proj, margin=self.triplet_margin
            )

        return w_s, loss_style


# ---------------------------
# MemoryTalker Stage 2
# ---------------------------

class MemoryTalker(nn.Module):
    """
    MemoryTalker Stage 2: Audio-Guided Stylization.

    - Module & parameter names are aligned with Stage 1:
        E_aud, txt_proj, E_m, PPE, D_m, fusion_layer, output_proj, memory_net
    - Adds:
        style_encoder (SpeakingStyleEncoder)
        stylized memory retrieval using w_s and M_m.
    """
    def __init__(self, args):
        super(MemoryTalker, self).__init__()
        self.device = args.device
        self.dataset = args.dataset
        self.vertice_dim = args.vertice_dim
        self.feature_dim = args.feature_dim
        self.mem_slot = args.mem_slot

        # 1. Audio Encoder (Stage1와 동일 이름/구조)
        self.E_aud = HubertForCTC.from_pretrained("facebook/hubert-base-ls960")
        self.E_aud.freeze_feature_encoder()

        # Projection: logits(mem_slot) -> feature_dim
        self.txt_proj = nn.Linear(self.mem_slot, self.feature_dim)
        nn.init.xavier_uniform_(self.txt_proj.weight)

        # Motion Encoder (Stage2에서는 사용하지 않을 수 있지만, state_dict 호환을 위해 유지)
        self.E_m = nn.Linear(self.vertice_dim, self.feature_dim)
        nn.init.xavier_uniform_(self.E_m.weight)

        # Dropout 설정
        self.dropout = nn.Dropout(0.0)

        # 2. Motion Decoder (Stage1와 동일 구조/이름)
        self.PPE = PeriodicPositionalEncoding(self.feature_dim, period=args.period)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.feature_dim,
            nhead=4,
            dim_feedforward=2 * self.feature_dim,
            batch_first=True,
        )
        self.D_m = nn.TransformerDecoder(decoder_layer, num_layers=1)

        # Fusion Layer: concat([f_txt, f_m_key]) -> feature_dim
        self.fusion_layer = nn.Linear(self.feature_dim * 2, self.feature_dim)
        nn.init.xavier_uniform_(self.fusion_layer.weight)

        # Output Projection: feature_dim -> vertice_dim
        self.output_proj = nn.Linear(self.feature_dim, self.vertice_dim)
        nn.init.constant_(self.output_proj.weight, 0.0)
        nn.init.constant_(self.output_proj.bias, 0.0)

        # 3. Memory Module (Stage1와 동일)
        self.memory_net = FacialMotionMemory(args)

        # 4. Style Encoder (Stage2 전용)
        self.style_encoder = SpeakingStyleEncoder(args)

    def forward(
        self,
        audio,
        template,
        vertice,
        one_hot,
        inference=False,
        audio_neg=None,
        audio_pos=None,
        lip_indices=None,
    ):
        """
        Returns:
            pred_motion: (B, T, V*3)
            loss_mse:    L_mse
            loss_vel:    L_vel
            loss_mem:    (Stage2에서는 0)
            loss_align:  (Stage2에서는 0)
            loss_style:  Triplet Style Loss
            loss_lip:    Lip Vertex Loss
        """
        # 초기 loss 텐서
        loss_mse = torch.tensor(0.0, device=self.device)
        loss_vel = torch.tensor(0.0, device=self.device)
        loss_lip = torch.tensor(0.0, device=self.device)
        loss_mem = torch.tensor(0.0, device=self.device)    # Stage2에서는 사용 X
        loss_align = torch.tensor(0.0, device=self.device)  # Stage2에서는 사용 X

        # Template: (B, V*3) -> (B, 1, V*3)
        template = template.unsqueeze(1)

        # 1. Audio -> Text logits
        if not inference:
            frame_num = vertice.shape[1]
            txt_logits = self.E_aud(audio, self.dataset, frame_num=frame_num).logits
        else:
            txt_logits = self.E_aud(audio, self.dataset).logits
            frame_num = txt_logits.shape[1]
            if self.dataset == "BIWI":
                frame_num //= 2

        # 2. Text representation
        f_txt = self.txt_proj(txt_logits)  # (B, T, c)

        # 3. Style Weights (w_s) & L_style
        w_s, loss_style = self.style_encoder(audio, audio_neg, audio_pos)
        # w_s: (B, mem_slot) -> 전역 slot 스케일로 사용 (배치 평균)
        if w_s.dim() == 2:
            w_s_slot = w_s.mean(dim=0)  # (mem_slot,)
        else:
            w_s_slot = w_s

        # Stylized Memory: M_m_tilde (n, c)
        # M_m: (n, c), w_s_slot: (n,)
        M_m_tilde = self.memory_net.M_m * w_s_slot.unsqueeze(-1)

        # 4. Stylized Retrieval
        if self.dataset == "BIWI":
            f_txt_interp = linear_interpolation(f_txt, 50, 25, frame_num)
            logits_interp = linear_interpolation(txt_logits, 50, 25, frame_num)

            # Key Address from text logits (Eq 5)
            K_txt = F.softmax(
                self.memory_net.scaling_term * logits_interp, dim=-1
            )  # (B, T, n)

            # Stylized motion retrieval
            f_m_key = self.memory_net.retrieve(K_txt, M_m_tilde)  # (B, T, c)

            fusion_input = torch.cat([f_txt_interp, f_m_key], dim=-1)
            decoder_mask = get_mask(
                self.device, self.dataset, logits_interp.shape[1], txt_logits.shape[1]
            )
        else:
            # VOCASET
            K_txt = F.softmax(
                self.memory_net.scaling_term * txt_logits, dim=-1
            )  # (B, T, n)
            f_m_key = self.memory_net.retrieve(K_txt, M_m_tilde)

            fusion_input = torch.cat([f_txt, f_m_key], dim=-1)
            decoder_mask = get_mask(
                self.device, self.dataset, txt_logits.shape[1], txt_logits.shape[1]
            )

        # 5. Decode
        proj_fusion = self.PPE(self.fusion_layer(fusion_input))
        decoded_feat = self.D_m(proj_fusion, f_txt, memory_mask=decoder_mask)
        residual_motion = self.output_proj(decoded_feat)
        pred_motion = residual_motion + template  # (B, T, V*3)

        # 6. Losses (Stage2)
        if not inference and vertice is not None:
            # L_mse
            loss_mse = F.mse_loss(pred_motion, vertice.detach())

            # L_vel
            loss_vel = compute_velocity_loss(
                pred_motion, vertice.detach(), reduction="mean"
            )

            # Lip Vertex Loss
            if lip_indices is not None:
                B, T, _ = pred_motion.shape
                pred_reshaped = pred_motion.reshape(B, T, -1, 3)
                gt_reshaped = vertice.reshape(B, T, -1, 3)

                loss_lip = F.mse_loss(
                    pred_reshaped[:, :, lip_indices, :],
                    gt_reshaped[:, :, lip_indices, :].detach(),
                )

        return (
            pred_motion,
            loss_mse,
            loss_vel,
            loss_mem,
            loss_align,
            loss_style,
            loss_lip,
        )