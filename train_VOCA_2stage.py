"""
Paper: MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization (ICCV 2025 Submission)
File: train_stage2.py

Description: 
Main training script for Stage 2 (Animating / Stylization).
This script loads the pre-trained Stage 1 model, freezes general motion parameters,
and trains the Speaking Style Encoder using Triplet Loss.
"""

import os
import shutil
import time
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

# Custom Modules
# Note: Ensure 'data_loader_contrastive.py' exists for triplet sampling
from data_loader_contrastive import get_dataloaders 
from models.MemoryTalker_2stage import MemoryTalker


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def trainer(args, train_loader, dev_loader, model, optimizer, criterion, lip_indices, start_epoch=1):
    save_dir = os.path.join(args.dataset, args.save_path)
    
    # Initialize Saving Directory
    if start_epoch == 1:
        if os.path.exists(save_dir): shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        best_epoch = 0
        total_lve_loss = np.inf
    else:
        # Resume Logic
        checkpoint_path = os.path.join(save_dir, f'{start_epoch-1}_model.pth')
        model.load_state_dict(torch.load(checkpoint_path))
        best_epoch = start_epoch - 1
        total_lve_loss = np.inf 

    # --- Training Loop ---
    for e in range(start_epoch, args.max_epoch + 1):
        loss_log = []
        model.train()
        
        # DataLoader in Stage 2 returns Triplet: (Anchor, Pos, Neg) for Audio
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Stage 2 Epoch {e}")
        optimizer.zero_grad()

        for i, batch_data in pbar:
            # Unpack Contrastive Batch
            # (audio, vertice, template, one_hot, file_name, audio_neg, audio_pos)
            audio_anchor, target_motion, template, one_hot, _, audio_neg, audio_pos = batch_data
            
            # To GPU
            audio_anchor = audio_anchor.to(args.device)
            audio_pos = audio_pos.to(args.device)
            audio_neg = audio_neg.to(args.device)
            target_motion = target_motion.to(args.device)
            template = template.to(args.device)
            one_hot = one_hot.to(args.device)

            # Forward Pass
            # outputs: (pred, L_mse, L_vel, 0, 0, L_style, L_lip)
            outputs = model(
                audio_anchor, template, target_motion, one_hot, 
                inference=False, 
                audio_neg=audio_neg, 
                audio_pos=audio_pos,
                lip_indices=lip_indices # Pass lip indices for internal Lip Loss calculation
            )
            
            loss_mse = outputs[1]       # L_mse
            loss_vel = outputs[2]       # L_vel
            loss_style = outputs[5]     # L_style (Triplet Loss, Eq 13)
            loss_lip = outputs[6]       # L_lip (Lip Vertex Loss)

            # Total Loss (Eq 17)
            # L = L_mse + L_vel + lambda2 * (L_lip + L_style)
            # Default lambda2 = 0.01
            total_loss = loss_mse + loss_vel + 0.01 * (loss_lip + loss_style)

            try:
                total_loss.backward()
            except RuntimeError as error:
                print(f"Gradient Error: {error}")
                continue

            loss_log.append(total_loss.item())
               
            if i % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_description(
                f"(Ep {e}) Loss:{np.mean(loss_log)*1e5:.2f} | "
                f"MSE:{loss_mse.item()*1e5:.2f} | "
                f"Style:{loss_style.item()*1e5:.2f} | "
                f"Lip:{loss_lip.item()*1e5:.2f}"
            )
            
        # --- Validation Loop ---
        valid_lve_log = []
        model.eval()
        
        with torch.no_grad():
            for audio, vertice, template, one_hot_all, _ in dev_loader:
                audio, vertice, template = audio.to(args.device), vertice.to(args.device), template.to(args.device)
                one_hot_all = one_hot_all.to(args.device)
                
                # Validate on first speaker identity for speed
                one_hot = one_hot_all[:, 0, :]
                
                pred_motion, _, _, _, _, _, _ = model(
                    audio, template, vertice, one_hot, inference=True
                )
                
                # Calculate LVE (Lip Vertex Error)
                min_len = min(pred_motion.shape[1], vertice.shape[1])
                pred_lip = pred_motion[:, :min_len].reshape(-1, min_len, args.vertice_dim//3, 3)[:, :, lip_indices, :]
                gt_lip = vertice[:, :min_len].reshape(-1, min_len, args.vertice_dim//3, 3)[:, :, lip_indices, :]
                
                lve_val = criterion(pred_lip, gt_lip)
                valid_lve_log.append(lve_val.item())

        current_lve = np.mean(valid_lve_log)
        print(f"Epoch {e} Validation LVE: {current_lve*1e5:.4f}")

        # Save Best Model
        if current_lve < total_lve_loss:
            total_lve_loss = current_lve
            best_epoch = e
            torch.save(model.state_dict(), os.path.join(save_dir, f'{best_epoch}_model.pth'))
            print(f"--> Best model saved (LVE: {total_lve_loss*1e5:.4f})")
        
        if e % 25 == 0:
             torch.save(model.state_dict(), os.path.join(save_dir, f'ckpt_{e}.pth'))

    return model, best_epoch

@torch.no_grad()
def test(args, model, test_loader, best_epoch):
    result_dir = os.path.join(args.dataset, args.result_path)
    if os.path.exists(result_dir): shutil.rmtree(result_dir)
    os.makedirs(result_dir)

    # Load Best Model
    model_path = os.path.join(args.dataset, args.save_path, f'{best_epoch}_model.pth')
    model.load_state_dict(torch.load(model_path))
    model = model.to(args.device)
    model.eval()

    total_time = 0
    total_frames = 0

    print(f"Starting Inference with model epoch {best_epoch}...")
    for audio, vertice, template, _, file_name in tqdm(test_loader, desc="Inference"):
        audio, template = audio.to(args.device), template.to(args.device)
        
        start = time.time()
        # Inference mode: No triplet audio needed
        prediction, _, _, _, _, _, _ = model(audio, template, None, None, inference=True)
        end = time.time()

        prediction = prediction.squeeze().cpu().numpy()
        num_frames = prediction.shape[0] if len(prediction.shape) > 1 else 1
        
        total_time += (end - start)
        total_frames += num_frames

        np.save(os.path.join(result_dir, file_name[0].replace(".wav", ".npy")), prediction)

    print(f"Inference Speed: {total_frames / total_time:.2f} FPS")

def main():
    parser = argparse.ArgumentParser(description='MemoryTalker Stage 2 Training')
    
    # Path & Data
    parser.add_argument("--root_path", type=str, required=True, help='Root project path')
    parser.add_argument("--dataset", type=str, default="vocaset", choices=["vocaset", "BIWI"])
    parser.add_argument("--wav_path", type=str, default="wav")
    parser.add_argument("--vertices_path", type=str, default="vertices_npy")
    parser.add_argument("--template_file", type=str, default="templates.pkl")
    
    # Training Params
    parser.add_argument("--exp_name", type=str, default="stylized", help='Experiment name')
    parser.add_argument("--lr", type=float, default=0.00005, help='Lower LR for Stage 2')
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")

    # Stage 2 Specifics
    parser.add_argument("--pretrained_model_path", type=str, required=True, help='Path to Stage 1 checkpoint')
    parser.add_argument("--triplet_margin", type=float, default=5.0, help='Margin for Triplet Loss')
    parser.add_argument("--softmax_scaling_term", type=float, default=16.0)
    
    # Model Params
    parser.add_argument("--vertice_dim", type=int, default=5023*3)
    parser.add_argument("--feature_dim", type=int, default=96)
    parser.add_argument("--mem_slot", type=int, default=32)
    parser.add_argument("--period", type=int, default=30)

    # Subjects (Keep defaults or load from file)
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA")

    args = parser.parse_args()

    args.save_path = f"save_{args.exp_name}_stage2"
    args.result_path = f"result_{args.exp_name}_stage2"

    # Load Lip Indices
    lip_path = os.path.join(args.root_path, args.dataset, "range/lips_coordinates.npy")
    lip_indices = np.load(lip_path) if os.path.exists(lip_path) else np.arange(100)

    # Build Model
    model = MemoryTalker(args)
    model.to(args.device)

    # Load Stage 1 Weights & Freeze
    print(f"Loading Stage 1 weights from {args.pretrained_model_path}...")
    state_dict = torch.load(args.pretrained_model_path)
    model.load_state_dict(state_dict, strict=False) # strict=False allows adding new Style Encoder params

    # Freeze General Motion Components
    for name, param in model.named_parameters():
        param.requires_grad = False # Default freeze
        
        # Unfreeze Style Encoder
        if "style_encoder" in name:
            param.requires_grad = True
        # Unfreeze Memory (if we want to adapt it, but paper says M_m is stylized, usually fixed base)
        # Based on user code: "for param in model.identity_encoder.parameters(): param.requires_grad = True"
        # So we only train style_encoder.

    print(f"Trainable Parameters (Stage 2): {count_parameters(model)}")

    dataset = get_dataloaders(args)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.MSELoss()

    model, best_epoch = trainer(args, dataset["train"], dataset["test"], model, optimizer, criterion, lip_indices, start_epoch=args.start_epoch)
    test(args, model, dataset["test"], best_epoch)

if __name__=="__main__":
    main()