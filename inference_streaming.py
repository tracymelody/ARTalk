#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import json
import torch
import argparse
import torchaudio
import numpy as np
import gradio as gr
from gtts import gTTS
from tqdm import tqdm
import math
import torch.nn.functional as F
import time # Import the time module

from app import BitwiseARModel # Assuming BitwiseARModel is in app/models.py
from app.flame_model import FLAMEModel, RenderMesh
from app.utils_videos import write_video

class ARTAvatarInferEngine:
    def __init__(self, load_gaga=False, fix_pose=False, clip_length=750, device='cuda'):
        self.device = device
        self.fix_pose = fix_pose
        self.clip_length = clip_length
        
        # Define the model variant tag
        model_variant_tag = 'wav2vec_mini' 
        # This assumes the underlying audio encoder type for BitwiseARModel is still 'wav2vec'
        underlying_audio_encoder_type = 'wav2vec'

        ckpt_path = f'./assets/ARTalk_{model_variant_tag}.pt' # Expected: ./assets/ARTalk_wav2vec_mini.pt
        config_path = f"./assets/config_mini.json"
        
        print(f"Loading checkpoint from: {ckpt_path}")
        print(f"Loading config from: {config_path}")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        configs = json.load(open(config_path))
        
        # Ensure the AR_CONFIG.AUDIO_ENCODER in the loaded config is set,
        # as BitwiseARModel uses this to select the audio encoder module.
        # If config_mini.json already has this correctly (e.g., "wav2vec"), this line is redundant
        # but ensures it if the key is missing or different than what the model structure expects.
        configs['AR_CONFIG']['AUDIO_ENCODER'] = underlying_audio_encoder_type 
        
        self.ARTalk = BitwiseARModel(configs).eval().to(device)
        self.ARTalk.load_state_dict(ckpt, strict=True)
        self.flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=1.0, no_lmks=True).to(device)
        self.mesh_renderer = RenderMesh(image_size=512, faces=self.flame_model.get_faces(), scale=1.0)
        
        # Modified output directory for the new model streaming version
        self.output_dir = f'render_results/ARTAvatar_{model_variant_tag}_streaming'
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        self.style_motion = None

        if load_gaga:
            from app.GAGAvatar import GAGAvatar
            self.GAGAvatar = GAGAvatar().to(device)
            self.GAGAvatar_flame = FLAMEModel(n_shape=300, n_exp=100, scale=5.0, no_lmks=True).to(device)

    def set_style_motion(self, style_motion):
        if isinstance(style_motion, str):
            style_motion_path = 'assets/style_motion/{}.pt'.format(style_motion)
            print(f"Loading style motion from: {style_motion_path}")
            style_motion = torch.load(style_motion_path, map_location='cpu', weights_only=True)
        assert style_motion.shape == (50, 106), f'Invalid style_motion shape: {style_motion.shape}. Expected (50, 106).'
        self.style_motion = style_motion[None].to(self.device)

    def inference(self, audio, clip_length=None):
        """
        Performs inference using the streaming API of BitwiseARModel.
        """
        batch_size = 1 # BitwiseARModel streaming methods expect batch_size=1
        self.ARTalk.inference_streaming_start(batch_size=batch_size, style_motion=self.style_motion)

        expected_chunk_len = self.ARTalk.expected_audio_chunk_length
        # For a 1-second model (patch_nums[-1]=25), expected_chunk_len should be 16000.
        print(f"Using expected audio chunk length: {expected_chunk_len} samples per chunk.")
        
        # Ensure audio is 1D (num_samples)
        if audio.ndim > 1:
            audio_flat = audio.mean(dim=0) 
        else:
            audio_flat = audio.clone() # Use clone to avoid modifying original tensor if it's passed around

        audio_chunks = list(audio_flat.split(expected_chunk_len))
        
        all_pred_motions_chunks = []
        total_chunk_processing_time = 0
        num_chunks_processed = 0

        print('Inferring motion (streaming)...')
        for i, chunk in enumerate(tqdm(audio_chunks, desc="Processing audio chunks")):
            current_chunk_len = chunk.shape[0]
            chunk_to_process = chunk
            # print(f"Current chunk length: {current_chunk_len}, Expected chunk length: {expected_chunk_len}") # Optional: for debugging chunk lengths
            if current_chunk_len < expected_chunk_len:
                padding_needed = expected_chunk_len - current_chunk_len
                chunk_to_process = F.pad(chunk, (0, padding_needed), "constant", 0)
            
            # Add batch dimension and send to device
            chunk_batch = chunk_to_process.unsqueeze(0).to(self.device)
            
            if self.device == 'cuda':
                torch.cuda.synchronize() # Ensure prior CUDA ops are done
            start_time = time.time()
            
            pred_motion_chunk = self.ARTalk.inference_streaming_chunk(chunk_batch)
            
            if self.device == 'cuda':
                torch.cuda.synchronize() # Ensure current CUDA op is done
            end_time = time.time()
            
            chunk_processing_time = end_time - start_time
            total_chunk_processing_time += chunk_processing_time
            num_chunks_processed += 1
            print(f"Chunk {i+1}/{len(audio_chunks)} processed in {chunk_processing_time:.4f} seconds.")
            
            all_pred_motions_chunks.append(pred_motion_chunk)

        self.ARTalk.inference_streaming_end()

        if num_chunks_processed > 0:
            average_time_per_chunk = total_chunk_processing_time / num_chunks_processed
            print(f"Average processing time per chunk: {average_time_per_chunk:.4f} seconds.")
        else:
            print("No chunks were processed.")


        if not all_pred_motions_chunks:
            print("Warning: No motion predicted from streaming, possibly due to very short audio.")
            # Assuming motion features are 106 from self.style_motion example
            return torch.empty(0, 106 if self.style_motion is None else self.style_motion.shape[-1], device=self.device) 

        # Concatenate results from all chunks (each is B, Frames, Features)
        pred_motions = torch.cat(all_pred_motions_chunks, dim=1)[0] # Remove batch dim -> (TotalFrames, Features)

        # Truncate to the number of frames corresponding to the original audio length
        # This matches the seq_length calculation in BitwiseARModel.inference
        total_expected_frames = math.ceil(audio_flat.shape[0] / 16000 * 25.0)
        pred_motions = pred_motions[:int(total_expected_frames)]

        # Apply smoothing
        pred_motions = self.smooth_motion_savgol(pred_motions)
        
        # Apply user-defined clip_length
        current_clip_length = clip_length if clip_length is not None else self.clip_length
        pred_motions = pred_motions[:current_clip_length]

        if self.fix_pose:
            pred_motions[..., 100:103] *= 0.0
        
        pred_motions[..., 104:] *= 0.0 # Zero out certain features as in original
        print('Done inferring motion (streaming)!')
        return pred_motions

    def rendering(self, audio, pred_motions, shape_id="mesh", shape_code=None, save_name='ARTAvatar_streaming.mp4'):
        print('Rendering...')
        pred_images = []
        if pred_motions.shape[0] == 0:
            print("No motion frames to render.")
            return

        if shape_id == "mesh":
            if shape_code is None:
                shape_code = torch.zeros(1, 300, device=self.device).expand(pred_motions.shape[0], -1)
            else:
                assert shape_code.dim() == 2, f'Invalid shape_code dim: {shape_code.dim()}.'
                assert shape_code.shape[0] == 1, f'Invalid shape_code shape: {shape_code.shape}.'
                shape_code = shape_code.to(self.device).expand(pred_motions.shape[0], -1)
            verts = self.ARTalk.basic_vae.get_flame_verts(self.flame_model, shape_code, pred_motions, with_global=True)
            for v in tqdm(verts, desc="Rendering frames"):
                rgb = self.mesh_renderer(v[None])[0]
                pred_images.append(rgb.cpu()[0] / 255.0)
        else:
            # assert isinstance(shape_id, str), f'Invalid shape_id type: {type(shape_id)}'
            self.GAGAvatar.set_avatar_id(shape_id)
            for motion in tqdm(pred_motions, desc="Rendering frames"):
                batch = self.GAGAvatar.build_forward_batch(motion[None], self.GAGAvatar_flame)
                rgb = self.GAGAvatar.forward_expression(batch)
                pred_images.append(rgb.cpu()[0])
        print('Done rendering!')
        
        if not pred_images:
            print("No images were rendered.")
            return

        # save video
        print('Saving video...')
        pred_images_tensor = torch.stack(pred_images)
        dump_path = os.path.join(self.output_dir, '{}.mp4'.format(save_name))
        
        # Ensure audio length matches video length
        max_audio_samples = int(pred_images_tensor.shape[0] / 25.0 * 16000)
        audio_for_video = audio[:max_audio_samples]
        
        write_video(pred_images_tensor*255.0, dump_path, 25.0, audio_for_video, 16000, "aac")
        print(f'Video saved to {dump_path}')
        print('Done saving video!')

    @staticmethod
    def smooth_motion_savgol(motion_codes):
        if motion_codes.shape[0] < 5: # Savgol filter needs enough data points
            print("Not enough frames to apply full Savgol smoothing.")
            if motion_codes.shape[0] < 2: return motion_codes # Cannot smooth if less than 2 frames
            # Apply simpler smoothing or skip if too few frames
            window_length_main = min(motion_codes.shape[0] if motion_codes.shape[0] % 2 != 0 else motion_codes.shape[0]-1, 5)
            window_length_pose = min(motion_codes.shape[0] if motion_codes.shape[0] % 2 != 0 else motion_codes.shape[0]-1, 9)
            if window_length_main < 2 : return motion_codes # Cannot smooth if window is too small
        else:
            window_length_main = 5
            window_length_pose = 9
            
        from scipy.signal import savgol_filter
        motion_np = motion_codes.clone().detach().cpu().numpy()
        
        # Ensure polyorder is less than window_length
        polyorder_main = min(2, window_length_main -1)
        polyorder_pose = min(3, window_length_pose -1)

        motion_np_smoothed = savgol_filter(motion_np, window_length=window_length_main, polyorder=polyorder_main, axis=0)
        if motion_np.shape[1] > 102: # Check if pose indices are valid
             motion_np_smoothed[..., 100:103] = savgol_filter(motion_np[..., 100:103], window_length=window_length_pose, polyorder=polyorder_pose, axis=0)
        return torch.tensor(motion_np_smoothed).type_as(motion_codes)


def run_gradio_app(engine):
    def process_audio(input_type, audio_input, text_input, text_language, shape_id, style_id):
        if input_type == "Audio" and audio_input is None:
            gr.Warning("Please upload an audio file")
            return None, None
        if input_type == "Text" and (text_input is None or len(text_input.strip()) == 0):
            gr.Warning("Please input text content") 
            return None, None
        
        output_filename_base = "tts_output"
        if input_type == "Text":
            gtts_lang = {"English": "en", "中文": "zh", "日本語": "ja", "Deutsch": "de", "Français": "fr", "Español": "es"}
            tts = gTTS(text=text_input, lang=gtts_lang[text_language])
            # Save TTS output to the engine's output directory to avoid permission issues in some environments
            tts_path = os.path.join(engine.output_dir, f"{output_filename_base}.wav")
            tts.save(tts_path)
            audio_input_path = tts_path
        else:
            audio_input_path = audio_input
            output_filename_base = os.path.splitext(os.path.basename(audio_input_path))[0]

        # load audio
        audio, sr = torchaudio.load(audio_input_path)
        audio_resampled = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)
        
        # inference
        if style_id == "default":
            engine.style_motion = None
        else:
            engine.set_style_motion(style_id)
        
        pred_motions = engine.inference(audio_resampled)
        
        if pred_motions.shape[0] == 0:
            gr.Warning("Motion generation failed or resulted in zero frames.")
            return None, None

        # render
        save_name = f'{output_filename_base}_{style_id.replace(".", "_")}_{shape_id.replace(".", "_")}'
        engine.rendering(audio_resampled, pred_motions, shape_id=shape_id, save_name=save_name)
        
        # save pred_motions
        motion_save_path = os.path.join(engine.output_dir, '{}_motions.pt'.format(save_name))
        torch.save(pred_motions.float().cpu(), motion_save_path)
        
        video_output_path = os.path.join(engine.output_dir, '{}.mp4'.format(save_name))
        return video_output_path, motion_save_path

    if hasattr(engine, 'GAGAvatar'):
        all_gagavatar_id = list(engine.GAGAvatar.all_gagavatar_id.keys())
        all_gagavatar_id = sorted(all_gagavatar_id)
    else:
        all_gagavatar_id = []
    
    style_motion_dir = 'assets/style_motion'
    if os.path.exists(style_motion_dir):
        all_style_id = [os.path.basename(i) for i in os.listdir(style_motion_dir)]
        all_style_id = sorted([i.split('.')[0] for i in all_style_id if i.endswith('.pt')])
    else:
        print(f"Warning: Style motion directory not found: {style_motion_dir}")
        all_style_id = []

    with gr.Blocks(title="ARTalk (Streaming): Speech-Driven 3D Head Animation") as demo:
        gr.Markdown("""
            <center>
            <h1>ARTalk (Streaming Test): Speech-Driven 3D Head Animation via Autoregressive Model</h1>
            </center>
            This version uses **streaming inference**.
        """)
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input Audio & Text")
                input_type = gr.Radio(choices=["Audio", "Text"], value="Audio", label="Choose input type")
                audio_group = gr.Group()
                with audio_group:
                    audio_input_ui = gr.Audio(type="filepath", label="Input Audio")
                text_group = gr.Group(visible=False)
                with text_group:
                    text_input_ui = gr.Textbox(label="Input Text")
                    text_language_ui = gr.Dropdown(choices=["English", "中文", "日本語", "Deutsch", "Français", "Español"], value="English", label="Choose the language of the input text")
            with gr.Column():
                gr.Markdown("### Avatar Control")
                appearance_ui = gr.Dropdown(
                    choices=["mesh"] + all_gagavatar_id,
                    value="mesh", label="Choose the apperance of the speaker",
                )
                style_ui = gr.Dropdown(
                    choices=["default"] + all_style_id,
                    value="natural_0" if "natural_0" in all_style_id else "default", label="Choose the style of the speaker",
                )
            with gr.Column():
                gr.Markdown("### Generated Video")
                video_output_ui = gr.Video(autoplay=True)
                motion_output_ui = gr.File(label="motion sequence", file_types=[".pt"])
                
        inputs_ui = [input_type, audio_input_ui, text_input_ui, text_language_ui, appearance_ui, style_ui]
        btn = gr.Button("Generate (Streaming)")
        btn.click(fn=process_audio, inputs=inputs_ui, outputs=[video_output_ui, motion_output_ui])

        # Define examples (ensure paths are valid)
        demo_audio_path = "demo/eng1.wav" # Example, adjust if not present
        example_shape = "mesh"
        example_style = "natural_0" if "natural_0" in all_style_id else "default"
        if not os.path.exists(demo_audio_path): demo_audio_path = None

        if hasattr(engine, 'GAGAvatar'):
            examples = [
                ["Audio", demo_audio_path, None, None, example_shape, example_style],
                ["Text", None, "Hello, this is a streaming demo of ARTalk!", "English", example_shape, example_style],
            ] if demo_audio_path else [
                 ["Text", None, "Hello, this is a streaming demo of ARTalk!", "English", example_shape, example_style],
            ]
        else:
            examples = [
                 ["Audio", demo_audio_path, None, None, "mesh", example_style],
                 ["Text", None, "This is a streaming test.", "English", "mesh", example_style],
            ] if demo_audio_path else [
                 ["Text", None, "This is a streaming test.", "English", "mesh", example_style],
            ]
        gr.Examples(examples=examples, inputs=inputs_ui, outputs=[video_output_ui, motion_output_ui])


        def toggle_input(choice):
            if choice == "Audio":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)
        input_type.change(
            fn=toggle_input, inputs=[input_type], outputs=[audio_group, text_group]
        )

    demo.launch(share=True) # Use a different port


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    
    parser = argparse.ArgumentParser(description="ARTalk Streaming Inference Script")
    # Consider adding an argument for model_variant if you switch frequently
    # parser.add_argument('--model_variant', default='wav2vec_mini', type=str, help="Model variant to use (e.g., 'wav2vec', 'wav2vec_mini').")
    parser.add_argument('--audio_path', '-a', default=None, type=str, help="Path to the input audio file.")
    parser.add_argument('--clip_length', '-l', default=750, type=int, help="Maximum length of the output motion in frames.")
    parser.add_argument("--shape_id", '-i', default='mesh', type=str, help="Shape ID for rendering (e.g., 'mesh' or GAGAvatar ID).")
    parser.add_argument("--style_id", '-s', default='default', type=str, help="Style ID for the motion (e.g., 'default', 'natural_0').")
    parser.add_argument("--load_gaga", action='store_true', help="Load GAGAvatar model for rendering different appearances.")
    parser.add_argument("--fix_pose", action='store_true', help="Fix the head pose (neck rotation) to zero.")
    parser.add_argument("--run_app", action='store_true', help="Run the Gradio web application interface.")
    parser.add_argument("--device", default='cuda', type=str, help="Device to run the models on (e.g., 'cuda', 'cpu').")

    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == 'cuda':
        print("CUDA is not available, switching to CPU.")
        args.device = 'cpu'

    engine = ARTAvatarInferEngine(
        load_gaga=args.load_gaga, 
        fix_pose=args.fix_pose, 
        clip_length=args.clip_length,
        device=args.device
    )
    
    if args.run_app:
        print("Running Gradio application for streaming inference...")
        run_gradio_app(engine)
    elif args.audio_path:
        print(f"Processing audio file: {args.audio_path}")
        if not os.path.exists(args.audio_path):
            print(f"Error: Audio file not found at {args.audio_path}")
            exit(1)
            
        audio, sr = torchaudio.load(args.audio_path)
        audio_resampled = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0).to(args.device)

        if args.style_id != 'default':
            engine.set_style_motion(args.style_id)
        else:
            engine.style_motion = None # Explicitly set to None for default

        print("Starting streaming inference...")
        pred_motions = engine.inference(audio_resampled, clip_length=args.clip_length)
        
        if pred_motions.shape[0] > 0:
            output_base_name = os.path.splitext(os.path.basename(args.audio_path))[0]
            save_name = f'{output_base_name}_{args.style_id.replace(".", "_")}_{args.shape_id.replace(".", "_")}'
            
            print("Rendering output video...")
            engine.rendering(audio_resampled.cpu(), pred_motions.cpu(), shape_id=args.shape_id, save_name=save_name)
            
            motion_save_path = os.path.join(engine.output_dir, f'{save_name}_motions.pt')
            torch.save(pred_motions.float().cpu(), motion_save_path)
            print(f"Predicted motions saved to: {motion_save_path}")
        else:
            print("No motions were predicted. Skipping rendering and saving.")
        print("Processing finished.")
    else:
        print("No audio path provided and --run_app is not set. Exiting.")
        parser.print_help() 