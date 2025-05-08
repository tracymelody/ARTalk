#!/usr/bin/env python
# Script to run inference on audio file and convert output to npy

import os
import torch
import argparse
import torchaudio
import numpy as np
from inference import ARTAvatarInferEngine

def run_inference(audio_path, style_id="natural_0", clip_length=750, output_dir="./"):
    """Run inference on the audio file and save the motion PT file"""
    print(f"Running inference on {audio_path}...")
    
    # Initialize the inference engine
    engine = ARTAvatarInferEngine(load_gaga=False, fix_pose=False, clip_length=clip_length)
    
    # Load audio
    audio, sr = torchaudio.load(audio_path)
    audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)
    
    # Set style and run inference
    if style_id != "default":
        engine.set_style_motion(style_id)
    
    # Get motion sequence
    pred_motions = engine.inference(audio)
    
    # Get base filename
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Save PT file
    pt_file_path = os.path.join(output_dir, f"{base_name}_motion.pt")
    torch.save(pred_motions.float().cpu(), pt_file_path)
    
    print(f"Saved motion sequence to {pt_file_path}")
    return pt_file_path

def convert_pt_to_npy(pt_file_path, output_dir="./"):
    """Convert PT motion file to NPY format"""
    print(f"Converting {pt_file_path} to NPY format...")
    
    # Load the PT file
    motion_data = torch.load(pt_file_path, map_location='cpu')
    
    # Convert to numpy array
    motion_np = motion_data.numpy()
    
    # Create output filename
    base_name = os.path.splitext(os.path.basename(pt_file_path))[0]
    npy_file_path = os.path.join(output_dir, f"{base_name}.npy")
    
    # Save as NPY file
    np.save(npy_file_path, motion_np)
    
    print(f"Converted PT file to NPY: {npy_file_path}")
    print(f"NPY shape: {motion_np.shape}, dtype: {motion_np.dtype}")
    
    return npy_file_path

def main():
    parser = argparse.ArgumentParser(description="Run inference on audio file and convert output to NPY")
    parser.add_argument("--audio_path", "-a", required=True, type=str, help="Path to the audio file")
    parser.add_argument("--style_id", "-s", default="natural_0", type=str, help="Style ID for motion generation")
    parser.add_argument("--clip_length", "-l", default=750, type=int, help="Maximum length of the output clip")
    parser.add_argument("--output_dir", "-o", default="./", type=str, help="Output directory")
    parser.add_argument("--skip_inference", action="store_true", help="Skip inference and just convert PT to NPY")
    parser.add_argument("--pt_file", "-p", type=str, help="Path to PT file (if skipping inference)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.skip_inference:
        # Run inference to get PT file
        pt_file_path = run_inference(
            args.audio_path, 
            args.style_id, 
            args.clip_length,
            args.output_dir
        )
    else:
        # Use provided PT file
        if not args.pt_file:
            raise ValueError("Must provide --pt_file when using --skip_inference")
        pt_file_path = args.pt_file
    
    # Convert PT to NPY
    npy_file_path = convert_pt_to_npy(pt_file_path, args.output_dir)
    
    print(f"Done! NPY file saved at: {npy_file_path}")

if __name__ == "__main__":
    main() 