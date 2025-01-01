import os
import torch
import random
import torchaudio
import folder_paths
import numpy as np
import platform
import subprocess
import sys
import importlib.util
import importlib.machinery
import argparse
from omegaconf import OmegaConf
import shutil

def import_inference_script(script_path):
    """Import a Python file as a module using its file path."""
    if not os.path.exists(script_path):
        raise ImportError(f"Script not found: {script_path}")
        
    module_name = "latentsync_inference"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None:
        raise ImportError(f"Failed to create module spec for {script_path}")
        
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        del sys.modules[module_name]
        raise ImportError(f"Failed to execute module: {str(e)}")
        
    return module

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

def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir


class LatentSyncNode:
    def __init__(self):
        check_and_install_dependencies()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "video_path": ("STRING", {"multiline": False, }),
                    "audio": ("AUDIO", ),
                    "seed" :("INT",{"default": 1247}),
                     },}

    CATEGORY = "LatentSyncNode"

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("video_path", )
    FUNCTION = "inference"

    def inference(self, video_path, audio, seed):
        cur_dir = get_ext_dir()
        ckpt_dir = os.path.join(cur_dir, "checkpoints")
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        # Normalize all paths
        video_path = normalize_path(video_path)
        if not os.path.exists(video_path):
            potential_video_path = normalize_path(os.path.join(folder_paths.get_input_directory(), video_path))
            if os.path.exists(potential_video_path):
                video_path = potential_video_path
            else:
                potential_video_path_with_ext = normalize_path(os.path.join(folder_paths.get_input_directory(), video_path + '.mp4'))
                if os.path.exists(potential_video_path_with_ext):
                    video_path = potential_video_path_with_ext
                else:
                    raise FileNotFoundError(f"Video file not found at: {video_path} or {potential_video_path} or {potential_video_path_with_ext}")

        if not os.path.exists(ckpt_dir):
            print("Downloading model checkpoints... This may take a while.")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="chunyu-li/LatentSync", 
                              allow_patterns=["latentsync_unet.pt", "whisper/tiny.pt"],
                              local_dir=ckpt_dir, local_dir_use_symlinks=False)
            print("Model checkpoints downloaded successfully!")
            
        inference_script_path = os.path.join(cur_dir, "scripts", "inference.py")
        unet_config_path = normalize_path(os.path.join(cur_dir, "configs", "unet", "second_stage.yaml"))
        scheduler_config_path = normalize_path(os.path.join(cur_dir, "configs"))
        ckpt_path = normalize_path(os.path.join(ckpt_dir, "latentsync_unet.pt"))
        whisper_ckpt_path = normalize_path(os.path.join(ckpt_dir, "whisper", "tiny.pt"))

        output_name = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        output_video_path = normalize_path(os.path.join(output_dir, f"latentsync_{output_name}_out.mp4"))

        # resample audio to 16k hz and save to wav
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        if waveform.dim() == 3: # Expected shape: [channels, samples]
            waveform = waveform.squeeze(0)

        if sample_rate != 16000:
            new_sample_rate = 16000
            waveform_16k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)(waveform)
            waveform, sample_rate = waveform_16k, new_sample_rate

        audio_path = normalize_path(os.path.join(output_dir, f"latentsync_{output_name}_audio.wav"))
        torchaudio.save(audio_path, waveform, sample_rate)

        print(f"Using video path: {video_path}")
        print(f"Video file exists: {os.path.exists(video_path)}")
        print(f"Video file size: {os.path.getsize(video_path)} bytes")
        
        assert os.path.exists(video_path), f"video_path not exists: {video_path}"
        assert os.path.exists(audio_path), f"audio_path not exists: {audio_path}"

        try:
            # Add the package root to Python path
            package_root = os.path.dirname(cur_dir)
            if package_root not in sys.path:
                sys.path.insert(0, package_root)
                
            # Add the current directory to Python path
            if cur_dir not in sys.path:
                sys.path.insert(0, cur_dir)

            # Import the inference module
            inference_module = import_inference_script(inference_script_path)
            
            # Create a Namespace object with the arguments
            args = argparse.Namespace(
                unet_config_path=unet_config_path,
                inference_ckpt_path=ckpt_path,
                video_path=video_path,
                audio_path=audio_path,
                video_out_path=output_video_path,
                seed=seed,
                scheduler_config_path=scheduler_config_path,
                whisper_ckpt_path=whisper_ckpt_path
            )
            
            # Load the config
            config = OmegaConf.load(unet_config_path)
            
            # Call main with both config and args
            inference_module.main(config, args)
            
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

        return (output_video_path,)


NODE_CLASS_MAPPINGS = {
    "D_LatentSyncNode": LatentSyncNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "D_LatentSyncNode": "LatentSync Node",
}