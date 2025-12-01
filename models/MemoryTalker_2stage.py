"""
Paper: MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization (ICCV 2025 Submission)
File: models/memory_talker_stage2.py

Description:
Stage 2 (Animating / Stylization) model for MemoryTalker.

- Module / parameter names are aligned with Stage 1 (memory_talker.py)
  so that Stage 1 checkpoints can be loaded seamlessly.
- Adds SpeakingStyleEncoder and stylized motion memory on top of the Stage 1 backbone.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from HuBERT import HubertForCTC
from models.audio_encoder_for_style import SpeakerEncoder, extract_logmel_torchaudio_tensor


# ---------------------------
# Common utility functions (shared with Stage 1)
# ---------------------------

def linear_interpolation(features, input_fps, output_fps, output_len=None):
    """
    Linearly interpolate features along the temporal dimension to match a target FPS.

    Args:
        features (Tensor): (B, T, C) input features.
        input_fps (int): Original frame rate.
        output_fps (int): Target frame rate.
        output_len (int, optional): Output length. If None, computed from fps ratio.

    Returns:
        Tensor: Interpolated features of shape (B, T_out, C).
    """
    features = features.transpose(1, 2)  # (B, C, T)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(
        features, size=output_len, align_corners=True, mode="linear"
    )
    return output_features.transpose(1, 2)  # (B, T_out, C)


def compute_velocity_loss(gt, estimated, reduction='mean'):
    """
    Compute velocity loss to penalize temporal jitter.

    L_vel = ||(v_{t+1} - v_t) - (v'_{t+1} - v'_t)||^2

    Args:
        gt (Tensor): Ground-truth motion (B, T, D).
        estimated (Tensor): Predicted motion (B, T, D).
        reduction (str): Reduction mode for mse_loss.

    Returns:
        Tensor: Scalar velocity loss.
    """
    gt_velocity = gt[:, 1:, :] - gt[:, :-1, :]
    estimated_velocity = estimated[:, 1:, :] - estimated[:, :-1, :]
    return F.mse_loss(gt_velocity, estimated_velocity, reduction=reduction)


def get_mask(device, dataset, T, S):
    """
    Build a decoder memory mask for the Transformer decoder.

    Args:
        device (torch.device): Target device.
        dataset (str): Dataset name ("BIWI" or "vocaset").
        T (int): Target sequence length.
        S (int): Source sequence length.

    Returns:
        Tensor: Boolean mask of shape (T, S), where True means masked.
    """
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        # BIWI often uses 2x audio frames per motion frame
        for i in range(T):
            mask[i, i * 2:i * 2 + 2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask == 1).to(device=device)


# ---------------------------
# Positional Encoding (shared with Stage 1)
# ---------------------------

class PeriodicPositionalEncoding(nn.Module):
    """
    Periodic positional encoding (PPE) for temporal sequences.

    This repeats a sinusoidal encoding pattern with a given period.
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
        """
        Args:
            x (Tensor): Input of shape (B, T, D).

        Returns:
            Tensor: Positionally encoded input with same shape.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------
# Facial Motion Memory (shared with Stage 1)
# ---------------------------

class FacialMotionMemory(nn.Module):
    """
    Facial Motion Memory (M_m) as described in Section 3.1.

    - Stores general facial motion features.
    - Provides cosine-similarity-based addressing and retrieval.
    """

    def __init__(self, args):
        super(FacialMotionMemory, self).__init__()
        self.feature_dim = args.feature_dim  # c
        self.mem_slot = args.mem_slot       # n
        self.scaling_term = args.softmax_scaling_term  # κ

        # M_m: (n, c) memory slots for general motion.
        self.M_m = nn.Parameter(
            torch.Tensor(self.mem_slot, self.feature_dim), requires_grad=True
        )
        nn.init.xavier_normal_(self.M_m)

    def get_similarity(self, query, memory):
        """
        Compute cosine similarity between query and memory slots.

        Args:
            query (Tensor): (B, T, c) query features.
            memory (Tensor): (n, c) memory slots.

        Returns:
            Tensor: Attention weights (B, T, n).
        """
        norm_q = F.normalize(query, p=2, dim=-1)
        norm_m = F.normalize(memory, p=2, dim=-1)

        similarity = torch.einsum('btd,sd->bts', norm_q, norm_m)  # (B, T, n)
        attention_weights = F.softmax(self.scaling_term * similarity, dim=-1)
        return attention_weights

    def retrieve(self, weights, memory):
        """
        Retrieve features via weighted sum over memory slots.

        Args:
            weights (Tensor): (B, T, n) attention weights.
            memory (Tensor): (n, c) memory slots.

        Returns:
            Tensor: Retrieved features (B, T, c).
        """
        retrieved_features = torch.einsum('bts,sd->btd', weights, memory)
        return retrieved_features

    def update_loss(self, f_txt, f_m):
        """
        Stage 1 memory update losses: L_align (KL) and L_mem (reconstruction).

        This is kept for state_dict compatibility but is not used in Stage 2.

        Args:
            f_txt (Tensor): Text features used as memory keys.
            f_m (Tensor): Motion features used as memory values.

        Returns:
            Tuple[Tensor, Tensor]: (loss_align, loss_mem)
        """
        # Text-based addressing (K_txt)
        K_txt = F.softmax(self.scaling_term * f_txt.detach(), dim=-1)

        # Motion-based addressing (V_m)
        V_m = self.get_similarity(query=f_m.detach(), memory=self.M_m)

        # Alignment loss (KL divergence between K_txt and V_m)
        loss_align = F.kl_div(V_m.log(), K_txt, reduction='batchmean')

        # Memory reconstruction loss
        f_m_val = self.retrieve(weights=V_m, memory=self.M_m)
        loss_mem = F.mse_loss(f_m.detach(), f_m_val)

        return loss_align, loss_mem

    def forward(self, f_txt):
        """
        Stage 1 forward path: retrieve general motion given text features.

        Kept for compatibility; Stage 2 uses explicit retrieve() with stylized memory.

        Args:
            f_txt (Tensor): Text logits or features (B, T, n_slots).

        Returns:
            Tensor: Retrieved motion features (B, T, c).
        """
        K_txt = F.softmax(self.scaling_term * f_txt, dim=-1)
        f_m_key = self.retrieve(weights=K_txt, memory=self.M_m.detach())
        return f_m_key


# ---------------------------
# Speaking Style Encoder (Stage 2 only)
# ---------------------------

class SpeakingStyleEncoder(nn.Module):
    """
    SpeakingStyleEncoder

    - Extracts style features f_s from audio.
    - Produces style weights w_s over memory slots.
    - Provides a triplet loss for style metric learning.
    """

    def __init__(self, args):
        super().__init__()
        self.mem_slot = args.mem_slot
        self.triplet_margin = args.triplet_margin
        self.device = args.device

        # Audio encoder for style representation
        self.audio_encoder = SpeakerEncoder()
        self.audio_encoder.initialize_weights()

        # Map encoder output to memory slot dimension
        self.to_style_emb = nn.Linear(256, self.mem_slot)

        # Global scaling factor
        self.to_scale = nn.Linear(self.mem_slot, 1, bias=False)

        # MLP for style metric learning (triplet)
        self.style_mlp = nn.Sequential(
            nn.Linear(self.mem_slot, self.mem_slot),
            nn.ReLU(),
            nn.Linear(self.mem_slot, self.mem_slot),
        )
        self._init_weights()

    def _init_weights(self):
        """
        Initialize linear layers with Xavier and zero bias.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, audio_anchor, audio_neg=None, audio_pos=None):
        """
        Forward pass for style encoding and triplet loss.

        Args:
            audio_anchor (Tensor): Anchor audio waveform/batch.
            audio_neg (Tensor, optional): Negative sample audio.
            audio_pos (Tensor, optional): Positive sample audio.

        Returns:
            Tuple[Tensor, Tensor]:
                w_s (Tensor): Style weights over slots, shape (B, n_slots).
                loss_style (Tensor): Triplet margin loss.
        """
        # Anchor style embedding
        mel_anchor = extract_logmel_torchaudio_tensor(audio_anchor)
        f_s = self.to_style_emb(self.audio_encoder(mel_anchor))  # (B, n_slots)

        # Slot-wise style weights w_s
        scale_factor = self.to_scale(f_s)  # (B, 1)
        w_s = torch.sigmoid(f_s) * scale_factor  # (B, n_slots)

        # Triplet loss for style space
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

    - Uses the same module names as Stage 1:
        E_aud, txt_proj, E_m, PPE, D_m, fusion_layer, output_proj, memory_net
      so that Stage 1 checkpoints can be loaded directly.
    - Adds:
        style_encoder (SpeakingStyleEncoder)
        stylized memory M_m_tilde via style weights w_s.
    """

    def __init__(self, args):
        super(MemoryTalker, self).__init__()
        self.device = args.device
        self.dataset = args.dataset
        self.vertice_dim = args.vertice_dim
        self.feature_dim = args.feature_dim
        self.mem_slot = args.mem_slot

        # 1. Audio encoder (ASR backbone) - same name as Stage 1
        self.E_aud = HubertForCTC.from_pretrained("facebook/hubert-base-ls960")
        self.E_aud.freeze_feature_encoder()

        # Project ASR logits (mem_slot) to feature_dim
        self.txt_proj = nn.Linear(self.mem_slot, self.feature_dim)
        nn.init.xavier_uniform_(self.txt_proj.weight)

        # Motion encoder (kept for compatibility; not strictly needed in Stage 2)
        self.E_m = nn.Linear(self.vertice_dim, self.feature_dim)
        nn.init.xavier_uniform_(self.E_m.weight)

        self.dropout = nn.Dropout(0.0)

        # 2. Transformer-based motion decoder (same structure/name as Stage 1)
        self.PPE = PeriodicPositionalEncoding(self.feature_dim, period=args.period)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.feature_dim,
            nhead=4,
            dim_feedforward=2 * self.feature_dim,
            batch_first=True,
        )
        self.D_m = nn.TransformerDecoder(decoder_layer, num_layers=1)

        # Fuse [f_txt, f_m_key] into a single representation
        self.fusion_layer = nn.Linear(self.feature_dim * 2, self.feature_dim)
        nn.init.xavier_uniform_(self.fusion_layer.weight)

        # Project fused features to vertex space (residual motion)
        self.output_proj = nn.Linear(self.feature_dim, self.vertice_dim)
        nn.init.constant_(self.output_proj.weight, 0.0)
        nn.init.constant_(self.output_proj.bias, 0.0)

        # 3. Memory module (shared with Stage 1)
        self.memory_net = FacialMotionMemory(args)

        # 4. Style encoder for stylization
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
        Forward pass for Stage 2 stylization.

        Args:
            audio (Tensor): Input audio.
            template (Tensor): Neutral template vertices (B, V*3).
            vertice (Tensor or None): GT vertices (B, T, V*3). None in inference.
            one_hot (Tensor or None): Speaker ID vector (unused here, kept for API).
            inference (bool): If True, run in inference mode (no losses).
            audio_neg (Tensor, optional): Negative sample for triplet.
            audio_pos (Tensor, optional): Positive sample for triplet.
            lip_indices (ndarray or Tensor, optional): Indices for lip vertices.

        Returns:
            Tuple:
                pred_motion (Tensor): Predicted vertices (B, T, V*3)
                loss_mse (Tensor): MSE loss
                loss_vel (Tensor): Velocity loss
                loss_mem (Tensor): (unused in Stage 2, always 0)
                loss_align (Tensor): (unused in Stage 2, always 0)
                loss_style (Tensor): Triplet style loss
                loss_lip (Tensor): Lip vertex loss
        """
        # Initialize loss tensors
        loss_mse = torch.tensor(0.0, device=self.device)
        loss_vel = torch.tensor(0.0, device=self.device)
        loss_lip = torch.tensor(0.0, device=self.device)
        loss_mem = torch.tensor(0.0, device=self.device)    # Not used in Stage 2
        loss_align = torch.tensor(0.0, device=self.device)  # Not used in Stage 2

        # Template: (B, V*3) -> (B, 1, V*3)
        template = template.unsqueeze(1)

        # 1. Audio -> ASR logits
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

        # 3. Style weights and style loss
        w_s, loss_style = self.style_encoder(audio, audio_neg, audio_pos)

        # Use batch-averaged style weights for the memory slots
        if w_s.dim() == 2:
            w_s_slot = w_s.mean(dim=0)  # (mem_slot,)
        else:
            w_s_slot = w_s

        # Stylized memory: M_m_tilde = diag(w_s_slot) * M_m
        # M_m: (n, c), w_s_slot: (n,) -> (n, c)
        M_m_tilde = self.memory_net.M_m * w_s_slot.unsqueeze(-1)

        # 4. Stylized retrieval
        if self.dataset == "BIWI":
            f_txt_interp = linear_interpolation(f_txt, 50, 25, frame_num)
            logits_interp = linear_interpolation(txt_logits, 50, 25, frame_num)

            # Key addressing from ASR logits
            K_txt = F.softmax(
                self.memory_net.scaling_term * logits_interp, dim=-1
            )  # (B, T, n)

            # Retrieve stylized motion
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

        # 5. Decode motion from fused features
        proj_fusion = self.PPE(self.fusion_layer(fusion_input))
        decoded_feat = self.D_m(proj_fusion, f_txt, memory_mask=decoder_mask)
        residual_motion = self.output_proj(decoded_feat)
        pred_motion = residual_motion + template  # (B, T, V*3)

        # 6. Losses (only in training mode)
        if not inference and vertice is not None:
            # MSE loss on full vertices
            loss_mse = F.mse_loss(pred_motion, vertice.detach())

            # Velocity (temporal smoothness) loss
            loss_vel = compute_velocity_loss(
                pred_motion, vertice.detach(), reduction="mean"
            )

            # Lip-only reconstruction loss
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