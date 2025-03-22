import argparse
import importlib.machinery
import importlib.util
import math
import os
import platform
import random
import shutil
import subprocess
import sys

import torch
import torchaudio
from omegaconf import OmegaConf
from .scripts import inference as inference_module
import folder_paths
import torchvision.io as io



def check_ffmpeg():
    try:
        if platform.system() == "Windows":
            # Check if ffmpeg exists in PATH
            ffmpeg_path = shutil.which("ffmpeg.exe")
            if ffmpeg_path is None:
                # Look for ffmpeg in common locations
                possible_paths = [
                    os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "ffmpeg", "bin"),
                    os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "ffmpeg", "bin"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg", "bin"),
                ]
                for path in possible_paths:
                    if os.path.exists(os.path.join(path, "ffmpeg.exe")):
                        # Add to PATH
                        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
                        return True
                print("FFmpeg not found. Please install FFmpeg and add it to PATH")
                return False
            return True
        else:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg not found. Please install FFmpeg")
        return False

def check_and_install_dependencies():
    if not check_ffmpeg():
        raise RuntimeError("FFmpeg is required but not found")

    required_packages = [
        'omegaconf',
        'pytorch_lightning',
        'transformers',
        'accelerate',
        'huggingface_hub',
        'einops',
        'diffusers'
    ]

    def is_package_installed(package_name):
        return importlib.util.find_spec(package_name) is not None

    def install_package(package):
        python_exe = sys.executable
        try:
            subprocess.check_call([python_exe, '-m', 'pip', 'install', package],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {str(e)}")
            raise RuntimeError(f"Failed to install required package: {package}")

    for package in required_packages:
        if not is_package_installed(package):
            print(f"Installing required package: {package}")
            try:
                install_package(package)
            except Exception as e:
                print(f"Warning: Failed to install {package}: {str(e)}")
                raise

def normalize_path(path):
    """Normalize path to handle spaces and special characters"""
    return os.path.normpath(path).replace('\\', '/')

def get_ext_dir(subpath=None, mkdir=True):
    """Get extension directory path, optionally with a subpath"""
    # Get the directory containing this script
    dir = folder_paths.models_dir
    
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    
    return dir

def setup_models():
    """Setup and pre-download all required models."""
    # Use our global temp directory
    global modles_dir

    # Existing setup logic for LatentSync models
    modles_dir = get_ext_dir(subpath="LatentSync")
    ckpt_dir = os.path.join(modles_dir, "checkpoints")
    whisper_dir = os.path.join(ckpt_dir, "whisper")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(whisper_dir, exist_ok=True)

    # Create a temp_downloads directory in our system temp
    temp_downloads = os.path.join(modles_dir, "downloads")
    os.makedirs(temp_downloads, exist_ok=True)
    
    unet_path = os.path.join(ckpt_dir, "latentsync_unet.pt")
    whisper_path = os.path.join(whisper_dir, "tiny.pt")

    if not (os.path.exists(unet_path) and os.path.exists(whisper_path)):
        print("Downloading required model checkpoints... This may take a while.")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="ByteDance/LatentSync-1.5",
                             allow_patterns=["latentsync_unet.pt", "whisper/tiny.pt"],
                             local_dir=ckpt_dir, 
                             local_dir_use_symlinks=False,
                             cache_dir=temp_downloads)
            print("Model checkpoints downloaded successfully!")
        except Exception as e:
            print(f"Error downloading models: {str(e)}")
            print("\nPlease download models manually:")
            print("1. Visit: https://huggingface.co/ByteDance/LatentSync-1.5")
            print("2. Download: latentsync_unet.pt and whisper/tiny.pt")
            print(f"3. Place them in: {ckpt_dir}")
            print(f"   with whisper/tiny.pt in: {whisper_dir}")
            raise RuntimeError("Model download failed. See instructions above.")

class LatentSyncNode:
    def __init__(self):
        check_and_install_dependencies()
        setup_models()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "images": ("IMAGE",),
                    "audio": ("AUDIO", ),
                    "seed": ("INT", {"default": 1247}),
                    "lips_expression": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
                    "inference_steps": ("INT", {"default": 20, "min": 1, "max": 999, "step": 1}),
                    "vae_name": (folder_paths.get_filename_list("vae"), {"tooltip": "sd1.5 vae model"}),
                 },
        }

    CATEGORY = "LatentSyncNode"

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio") 
    FUNCTION = "inference"

    def process_batch(self, batch, use_mixed_precision=False):
        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            processed_batch = batch.float() / 255.0
            if len(processed_batch.shape) == 3:
                processed_batch = processed_batch.unsqueeze(0)
            if processed_batch.shape[0] == 3:
                processed_batch = processed_batch.permute(1, 2, 0)
            if processed_batch.shape[-1] == 4:
                processed_batch = processed_batch[..., :3]
            return processed_batch

    def inference(self, images, audio, seed, vae_name, lips_expression=1.5, inference_steps=20):
        # Get GPU capabilities and memory
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        BATCH_SIZE = 4
        use_mixed_precision = False
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            # Convert to GB
            gpu_mem_gb = gpu_mem / (1024 ** 3)

            # Dynamically adjust batch size based on GPU memory
            if gpu_mem_gb > 20:  # High-end GPUs
                BATCH_SIZE = 32
                enable_tf32 = True
                use_mixed_precision = True
            elif gpu_mem_gb > 8:  # Mid-range GPUs
                BATCH_SIZE = 16
                enable_tf32 = False
                use_mixed_precision = True
            else:  # Lower-end GPUs
                BATCH_SIZE = 8
                enable_tf32 = False
                use_mixed_precision = False

            # Set performance options based on GPU capability
            torch.backends.cudnn.benchmark = True
            if enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            # Clear GPU cache before processing
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.8)

        # Create a run-specific subdirectory in our temp directory
        run_id = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        temp_dir = os.path.join(folder_paths.get_temp_directory(), f"run_{run_id}")
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_video_path = None
        output_video_path = None
        audio_path = None

        try:
            # Create temporary file paths in our system temp directory
            temp_video_path = os.path.join(temp_dir, f"temp_{run_id}.mp4")
            output_video_path = os.path.join(temp_dir, f"latentsync_{run_id}_out.mp4")
            audio_path = os.path.join(temp_dir, f"latentsync_{run_id}_audio.wav")
            
            # Get the extension directory
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Process input frames
            if isinstance(images, list):
                frames = torch.stack(images).to(device)
            else:
                frames = images.to(device)
            frames = (frames * 255).byte()

            if len(frames.shape) == 3:
                frames = frames.unsqueeze(0)

            # Process audio with device awareness
            waveform = audio["waveform"].to(device)
            sample_rate = audio["sample_rate"]
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)

            if sample_rate != 16000:
                new_sample_rate = 16000
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=new_sample_rate
                ).to(device)
                waveform_16k = resampler(waveform)
                waveform, sample_rate = waveform_16k, new_sample_rate

            # Package resampled audio
            resampled_audio = {
                "waveform": waveform.unsqueeze(0),
                "sample_rate": sample_rate
            }
            
            # Move waveform to CPU for saving
            waveform_cpu = waveform.cpu()
            torchaudio.save(audio_path, waveform_cpu, sample_rate)

            # Move frames to CPU for saving to video
            frames_cpu = frames.cpu()
            try:
                io.write_video(temp_video_path, frames_cpu, fps=25, video_codec='h264')
            except TypeError:
                import av
                container = av.open(temp_video_path, mode='w')
                stream = container.add_stream('h264', rate=25)
                stream.width = frames_cpu.shape[2]
                stream.height = frames_cpu.shape[1]

                for frame in frames_cpu:
                    frame = av.VideoFrame.from_ndarray(frame.numpy(), format='rgb24')
                    packet = stream.encode(frame)
                    container.mux(packet)

                packet = stream.encode(None)
                container.mux(packet)
                container.close()

            # Define paths to required files and configs
            # inference_script_path = os.path.join(cur_dir, "scripts", "inference.py")
            config_path = os.path.join(cur_dir, "configs", "unet", "stage2.yaml")
            scheduler_config_path = os.path.join(cur_dir, "configs")
            ckpt_path = os.path.join(modles_dir, "checkpoints", "latentsync_unet.pt")
            whisper_ckpt_path = os.path.join(modles_dir, "checkpoints", "whisper", "tiny.pt")

            # Create config and args
            config = OmegaConf.load(config_path)

            # Set the correct mask image path
            mask_image_path = os.path.join(cur_dir, "latentsync", "utils", "mask.png")
            # Make sure the mask image exists
            if not os.path.exists(mask_image_path):
                # Try to find it in the utils directory directly
                alt_mask_path = os.path.join(cur_dir, "utils", "mask.png")
                if os.path.exists(alt_mask_path):
                    mask_image_path = alt_mask_path
                else:
                    print(f"Warning: Could not find mask image at expected locations")

            # Set mask path in config
            if hasattr(config, "data") and hasattr(config.data, "mask_image_path"):
                config.data.mask_image_path = mask_image_path

            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)

            args = argparse.Namespace(
                unet_config_path=config_path,
                inference_ckpt_path=ckpt_path,
                video_path=temp_video_path,
                audio_path=audio_path,
                video_out_path=output_video_path,
                seed=seed,
                inference_steps=inference_steps,
                guidance_scale=lips_expression,  # Using lips_expression for the guidance_scale
                scheduler_config_path=scheduler_config_path,
                whisper_ckpt_path=whisper_ckpt_path,
                device=device,
                batch_size=BATCH_SIZE,
                use_mixed_precision=use_mixed_precision,
                temp_dir=temp_dir,
                mask_image_path=mask_image_path,
                vae_model_path=vae_path
            )

            # Set PYTHONPATH to include our directories 
            # package_root = os.path.dirname(cur_dir)
            # if package_root not in sys.path:
            #     sys.path.insert(0, package_root)
            # if cur_dir not in sys.path:
            #     sys.path.insert(0, cur_dir)

            # Clean GPU cache before inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Import the inference module
            # inference_module = import_inference_script(inference_script_path)
            
            # Monkey patch any temp directory functions in the inference module
            if hasattr(inference_module, 'get_temp_dir'):
                inference_module.get_temp_dir = lambda *args, **kwargs: temp_dir
                
            # Create subdirectories that the inference module might expect
            inference_temp = os.path.join(temp_dir, "temp")
            os.makedirs(inference_temp, exist_ok=True)
            
            # Run inference
            inference_module.main(config, args)

            # Clean GPU cache after inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Verify output file exists
            if not os.path.exists(output_video_path):
                raise FileNotFoundError(f"Output video not found at: {output_video_path}")
            
            # Read the processed video - ensure it's loaded as CPU tensor
            processed_frames = io.read_video(output_video_path, pts_unit='sec')[0]
            processed_frames = processed_frames.float() / 255.0

            # Ensure audio is on CPU before returning
            if torch.cuda.is_available():
                if hasattr(resampled_audio["waveform"], 'device') and resampled_audio["waveform"].device.type == 'cuda':
                    resampled_audio["waveform"] = resampled_audio["waveform"].cpu()
                if hasattr(processed_frames, 'device') and processed_frames.device.type == 'cuda':
                    processed_frames = processed_frames.cpu()

            return processed_frames, resampled_audio

        except Exception as e:
            print(f"Error during inference: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

        finally:
            # Clean up temporary files individually
            for path in [temp_video_path, output_video_path, audio_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        print(f"Removed temporary file: {path}")
                    except Exception as e:
                        print(f"Failed to remove {path}: {str(e)}")

            # Remove temporary run directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    print(f"Removed run temporary directory: {temp_dir}")
                except Exception as e:
                    print(f"Failed to remove temp run directory: {str(e)}")

            # Final GPU cache cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

class VideoLengthAdjuster:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "mode": (["normal", "pingpong", "loop_to_audio"], {"default": "normal"}),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0}),
                "silent_padding_sec": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 3.0, "step": 0.1}),
            }
        }

    CATEGORY = "LatentSyncNode"
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "adjust"

    def adjust(self, images, audio, mode, fps=25.0, silent_padding_sec=0.5):
        waveform = audio["waveform"].squeeze(0)
        sample_rate = int(audio["sample_rate"])
        original_frames = [images[i] for i in range(images.shape[0])] if isinstance(images, torch.Tensor) else images.copy()

        if mode == "normal":
            # Bypass video frames exactly
            video_duration = len(original_frames) / fps
            required_samples = int(video_duration * sample_rate)
            
            # Adjust audio to match video duration
            if waveform.shape[1] >= required_samples:
                adjusted_audio = waveform[:, :required_samples]  # Trim audio
            else:
                silence = torch.zeros((waveform.shape[0], required_samples - waveform.shape[1]), dtype=waveform.dtype)
                adjusted_audio = torch.cat([waveform, silence], dim=1)  # Pad audio
            
            return (
                torch.stack(original_frames),
                {"waveform": adjusted_audio.unsqueeze(0), "sample_rate": sample_rate}
            )

        elif mode == "pingpong":
            video_duration = len(original_frames) / fps
            audio_duration = waveform.shape[1] / sample_rate
            if audio_duration <= video_duration:
                required_samples = int(video_duration * sample_rate)
                silence = torch.zeros((waveform.shape[0], required_samples - waveform.shape[1]), dtype=waveform.dtype)
                adjusted_audio = torch.cat([waveform, silence], dim=1)

                return (
                    torch.stack(original_frames),
                    {"waveform": adjusted_audio.unsqueeze(0), "sample_rate": sample_rate}
                )

            else:
                silence_samples = math.ceil(silent_padding_sec * sample_rate)
                silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
                padded_audio = torch.cat([waveform, silence], dim=1)
                total_duration = (waveform.shape[1] + silence_samples) / sample_rate
                target_frames = math.ceil(total_duration * fps)
                reversed_frames = original_frames[::-1][1:-1]  # Remove endpoints
                frames = original_frames + reversed_frames
                while len(frames) < target_frames:
                    frames += frames[:target_frames - len(frames)]
                return (
                    torch.stack(frames[:target_frames]),
                    {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
                )

        elif mode == "loop_to_audio":
            # Add silent padding then simple loop
            silence_samples = math.ceil(silent_padding_sec * sample_rate)
            silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
            padded_audio = torch.cat([waveform, silence], dim=1)
            total_duration = (waveform.shape[1] + silence_samples) / sample_rate
            target_frames = math.ceil(total_duration * fps)

            frames = original_frames.copy()
            while len(frames) < target_frames:
                frames += original_frames[:target_frames - len(frames)]
            
            return (
                torch.stack(frames[:target_frames]),
                {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
            )

# Node Mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "LatentSyncNode": LatentSyncNode,
    "VideoLengthAdjuster": VideoLengthAdjuster,
}

# Display Names for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentSyncNode": "LatentSync1.5 Node",
    "VideoLengthAdjuster": "Video Length Adjuster",
}
