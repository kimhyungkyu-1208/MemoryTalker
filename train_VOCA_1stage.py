"""
Paper: MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization (ICCV 2025 Submission)
Authors: Hyung Kyu Kim, Sangmin Lee, Hak Gu Kim
Institution: Chung-Ang University, Korea University

Description: Main training script for MemoryTalker.
This script handles both Stage 1 (Memorizing General Motion) and Stage 2 (Audio-Guided Stylization).
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
from data_loader import get_dataloaders
from models.MemoryTalker_1stage import MemoryTalker # Renamed for clarity


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def trainer(args, train_loader, dev_loader, model, optimizer, criterion, lip_indices, epoch=100, start_epoch=1):
    """
    Training loop for MemoryTalker.
    Handles distinct logic for Stage 1 and Stage 2 based on args.stage.
    """
    
    # Logging & Saving Setup
    save_dir = os.path.join(args.dataset, args.save_path)
    if start_epoch == 1:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        best_epoch = 0
        iteration = 0
        total_lve_loss = np.inf
    else:
        # Resume training
        checkpoint_path = os.path.join(save_dir, f'{start_epoch-1}_model.pth')
        print(f"Resuming training from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path))
        best_epoch = start_epoch - 1
        iteration = len(train_loader) * (start_epoch - 1)
        total_lve_loss = np.inf # Reset or load from log if available

    # --- Stage 2 Configuration: Audio-Guided Stylization ---
    # Reference: Section 4.2 Implementation Details
    # "In the second stage, we freeze all layers of the first stage and train only the speaking style encoder."
    if args.stage == 2:
        print(">>> Configuring for Stage 2: Audio-Guided Stylization")
        
        if args.pretrained_model_path is None:
            raise ValueError("Stage 2 requires --pretrained_model_path (Stage 1 checkpoint).")

        # Load Stage 1 weights
        print(f"Loading Stage 1 weights from {args.pretrained_model_path}")
        pretrained_state_dict = torch.load(args.pretrained_model_path)
        model.load_state_dict(pretrained_state_dict, strict=False)
        
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze Style Encoder / Identity Memory related parameters
        # TODO: Ensure 'identity_memory' or 'style_encoder' matches your model definition
        if hasattr(model, 'style_encoder'):
            for param in model.style_encoder.parameters():
                param.requires_grad = True
        elif hasattr(model, 'identity_memory'): # Legacy name fallback
            for param in model.identity_memory.parameters():
                param.requires_grad = True
        else:
            print("Warning: Could not find 'style_encoder' or 'identity_memory' to unfreeze.")

        # Re-initialize optimizer for active parameters only
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # --- Training Loop ---
    for e in range(start_epoch, epoch + 1):
        loss_log = []
        model.train()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {e}")
        optimizer.zero_grad()

        for i, (audio_input, target_motion, neutral_template, one_hot, file_name) in pbar:
            iteration += 1
            
            # Data to GPU (a: audio, v: facial motion)
            audio_input = audio_input.to(args.device)          # a
            target_motion = target_motion.to(args.device)      # v
            neutral_template = neutral_template.to(args.device)
            one_hot = one_hot.to(args.device)

            # Forward Pass
            # Expecting model to return losses based on Eq 11 (Stage 1) or Eq 17 (Stage 2)
            # Unpacking based on typical implementation. Adjust indices if model returns differently.
            # outputs: (pred_motion, loss_mse, loss_vel, loss_mem, loss_align, loss_style, loss_pair)
            outputs = model(audio_input, neutral_template, target_motion, one_hot, inference=False)
            
            estimated_motion = outputs[0]
            loss_mse = outputs[1]       # L_mse (Eq 9)
            loss_vel = outputs[2]       # L_vel (Eq 10)
            loss_mem = outputs[3]       # L_mem (Eq 3)
            loss_align = outputs[4]     # L_align (Eq 7)
            loss_style = outputs[5]     # L_style (Eq 13) - Active in Stage 2
            loss_pair = outputs[6]      # Auxiliary loss if any

            # Loss Composition
            # Stage 1: L = L_mse + L_vel + lambda1 * (L_mem + L_align)
            # Stage 2: L = L_mse + L_vel + lambda2 * (L_lip + L_style) (Lip loss usually calc in loop or model)
            
            # Using weights from args or defaults
            # Weights order: [MSE, Vel, Mem, Align, Style, Pair]
            # Standard paper config: lambda1=0.01, lambda2=0.01
            
            total_loss = 0
            if args.stage == 1:
                # Weights: [1, 1, 0.01, 0.01, 0, 0] roughly
                total_loss = loss_mse + loss_vel + 0.01 * (loss_mem + loss_align)
            else:
                # Stage 2
                # Weights: [1, 1, 0, 0, 0.01, 0] roughly + Lip Loss
                total_loss = loss_mse + loss_vel + 0.01 * loss_style 
                # Note: L_lip is typically calculated here or inside model. 
                # If inside model and part of mse/vel, this is fine.
                # If separate, add: + 0.01 * loss_lip 

            try:
                total_loss.backward()
            except RuntimeError as error:
                print(f"Gradient Error: {error}")
                continue

            loss_log.append(total_loss.item())
               
            if i % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Update Progress Bar
            pbar.set_description(
                f"(Ep {e}) "
                f"Loss:{np.mean(loss_log)*1e5:.2f} | "
                f"MSE:{loss_mse.item()*1e5:.2f} | "
                f"Mem:{loss_mem.item()*1e5:.2f} | "
                f"Align:{loss_align.item()*1e5:.2f} | "
                f"Style:{loss_style.item()*1e5:.2f}"
            )
            
        # --- Validation Loop ---
        valid_loss_log = []
        valid_lve_loss_log = []
        model.eval()
        
        with torch.no_grad():
            for audio_input, target_motion, neutral_template, one_hot_all, file_name in dev_loader:
                # Skip if invalid data
                if torch.all(one_hot_all == 0): continue
                
                audio_input = audio_input.to(args.device)
                target_motion = target_motion.to(args.device)
                neutral_template = neutral_template.to(args.device)
                one_hot_all = one_hot_all.to(args.device) # (B, Num_Speakers, D)

                # Validation usually runs on the first speaker or averages
                # Here we check the first iteration as in original code
                for iter_idx in range(one_hot_all.shape[-1]):
                    one_hot = one_hot_all[:, iter_idx, :]
                    
                    # Inference
                    estimated_motion, _, _, _, _, _, _ = model(
                        audio_input, neutral_template, target_motion, one_hot, inference=True
                    )
                    
                    # Align lengths
                    min_frame = min(estimated_motion.size(1), target_motion.size(1))
                    estimated_motion = estimated_motion[:, :min_frame]
                    target_motion = target_motion[:, :min_frame]

                    # Metrics
                    val_mse = criterion(estimated_motion, target_motion)
                    
                    # Calculate LVE (Lip Vertex Error) - Key Metric
                    # Reshape to (B, T, V, 3)
                    est_reshaped = estimated_motion.reshape(-1, min_frame, args.vertice_dim//3, 3)
                    gt_reshaped = target_motion.reshape(-1, min_frame, args.vertice_dim//3, 3)
                    
                    # Select Lip Region
                    val_lve = criterion(
                        est_reshaped[:, :, lip_indices, :], 
                        gt_reshaped[:, :, lip_indices, :]
                    )
                    
                    valid_loss_log.append(val_mse.item())
                    valid_lve_loss_log.append(val_lve.item())
                    
                    # For validation speed, check only first speaker style
                    if iter_idx == 0: break
                        
        current_val_loss = np.mean(valid_loss_log)
        current_lve_loss = np.mean(valid_lve_loss_log)
        
        print(f"Epoch {e} Summary: Val Loss: {current_val_loss*1e5:.4f}, Val LVE: {current_lve_loss*1e5:.4f}")

        # Regular Checkpoint
        if e % 25 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'checkpoint_{e}.pth'))
    
    return model, e

@torch.no_grad()
def test(args, model, test_loader, best_epoch):
    """
    Inference script to generate 3D facial animation sequences (.npy).
    """
    print(f"Starting Inference using epoch {best_epoch}...")
    result_dir = os.path.join(args.dataset, args.result_path)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)

    # Load Best Model
    save_dir = os.path.join(args.dataset, args.save_path)
    model_path = os.path.join(save_dir, f'checkpoint_{best_epoch}.pth')
    model.load_state_dict(torch.load(model_path))
    model = model.to(args.device)
    model.eval()

    total_time = 0
    total_frames = 0

    for audio_input, target_motion, neutral_template, one_hot_all, file_name in tqdm(test_loader, desc="Testing"):
        audio_input = audio_input.to(args.device)
        neutral_template = neutral_template.to(args.device)
        target_motion = target_motion.to(args.device) # Not used for generation, only reference

        start_time = time.time()
        
        # Inference: one_hot is None or learned internally in Stage 2
        # outputs[0] is prediction
        prediction, _, _, _, _, _, _ = model(
            audio_input, neutral_template, target_motion, None, inference=True
        )
        
        end_time = time.time()
        
        # Performance metrics
        prediction = prediction.squeeze() # (T, V*3)
        num_frames = prediction.shape[0] if prediction.dim() > 0 else 1
        total_time += (end_time - start_time)
        total_frames += num_frames

        # Save Result
        save_name = file_name[0].split(".")[0] + ".npy"
        np.save(os.path.join(result_dir, save_name), prediction.cpu().numpy())
    
    fps = total_frames / total_time if total_time > 0 else 0
    print(f"Inference complete. Average Speed: {fps:.2f} FPS")

def main():
    parser = argparse.ArgumentParser(description='MemoryTalker: Personalized Speech-Driven 3D Facial Animation')
    
    # Experiment Setup
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2], help='Training stage: 1 (Memorizing) or 2 (Stylization)')
    parser.add_argument("--exp_name", type=str, default="baseline", help='Experiment name for saving/logging')
    parser.add_argument("--start_epoch", type=int, default=1, help='Epoch to start/resume training from')
    parser.add_argument("--max_epoch", type=int, default=100, help='Total number of epochs')
    
    # Paths
    parser.add_argument("--root_path", type=str, required=True, help='Root path to project/data')
    parser.add_argument("--dataset", type=str, default="vocaset", choices=["vocaset", "BIWI"], help='Dataset name')
    parser.add_argument("--wav_path", type=str, default="wav", help='Subpath for audio')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='Subpath for vertices')
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='Subpath for templates')
    
    # Model Hyperparameters
    parser.add_argument("--vertice_dim", type=int, default=5023*3, help='Total vertex dimension (V*3)')
    parser.add_argument("--feature_dim", type=int, default=96, help='Motion feature dimension (d_m)')
    parser.add_argument("--mem_slot", type=int, default=32, help='Number of memory slots (n)')
    parser.add_argument("--softmax_scaling_term", type=float, default=16.0, help='Scaling factor for addressing')
    parser.add_argument("--lr", type=float, default=0.0001, help='Learning rate')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--period", type=int, default=30, help='Period in PPE (30 for VOCASET, 25 for BIWI)')
    
    # VOCASET Default Split
    parser.add_argument("--train_subjects", type=str, 
        default="FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA",
        help='Training subjects string separated by space')
    
    parser.add_argument("--val_subjects", type=str, 
        default="FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA",
        help='Validation subjects string separated by space')
    
    parser.add_argument("--test_subjects", type=str, 
        default="FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA",
        help='Test subjects string separated by space')
    
    # System
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Dynamic Path Setup
    args.save_path = f"save_{args.exp_name}_stage{args.stage}"
    args.result_path = f"result_{args.exp_name}_stage{args.stage}"
    
    # Load Lip Indices for LVE metric
    # Ensure this file exists at the specified path
    lip_path = os.path.join(args.root_path, args.dataset, "range/lips_coordinates.npy")
    if not os.path.exists(lip_path):
        print(f"Warning: Lip coordinates file not found at {lip_path}. LVE metric may be inaccurate.")
        # Dummy indices or exit
        lip_indices = np.arange(0, 100) 
    else:
        lip_indices = np.load(lip_path)

    # Initialize Model
    print(f"Initializing MemoryTalker (Stage {args.stage})...")
    memory_talker = MemoryTalker(args)
    print(f"Total Parameters: {count_parameters(memory_talker)}")

    if torch.cuda.is_available():
        memory_talker = memory_talker.to(args.device)

    # Load Data
    print("Loading Data...")
    dataloaders = get_dataloaders(args) # Expecting this to use args.root_path internally
    
    # Optimizer & Loss
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, memory_talker.parameters()), 
        lr=args.lr
    )
    criterion = nn.MSELoss()

    # Run Training
    memory_talker, best_epoch = trainer(
        args, 
        dataloaders["train"], 
        dataloaders["valid"], 
        memory_talker, 
        optimizer, 
        criterion, 
        lip_indices,
        epoch=args.max_epoch, 
        start_epoch=args.start_epoch
    )

    # Run Inference
    test(args, memory_talker, dataloaders["test"], best_epoch)

if __name__ == "__main__":
    main()