# ComfyUI-LatentSyncWrapper

Unofficial [LatentSync](https://github.com/bytedance/LatentSync) implementation for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) on windows.

This node provides lip-sync capabilities in ComfyUI using ByteDance's LatentSync model. It allows you to synchronize video lips with audio input.

![image](https://github.com/user-attachments/assets/678f5319-90b1-4c0a-b7ae-d3f01295157f)

https://github.com/user-attachments/assets/8e7ec7ad-ef88-4705-9899-495680360075

## Prerequisites

Before installing this node, you must install the following in order:

1. [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed and working

2. Required custom node:
   - [ComfyUI_AceNodes](https://github.com/hay86/ComfyUI_AceNodes) - Required for video in and out processing
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/hay86/ComfyUI_AceNodes
   cd ComfyUI_AceNodes
   pip install -r requirements.txt
   ```

3. FFmpeg installed on your system:
   - Windows: Download from [here](https://github.com/BtbN/FFmpeg-Builds/releases) and add to system PATH

## Installation

Only proceed with installation after confirming all prerequisites are installed and working.

1. Clone this repository into your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ShymuelRonen/ComfyUI-LatentSyncWrapper
cd ComfyUI-LatentSyncWrapper
pip install -r requirements.txt
```

## Required Dependencies
```
diffusers
transformers
huggingface-hub
omegaconf
einops
opencv-python
mediapipe
face-alignment
decord
ffmpeg-python
safetensors
soundfile
```

## Usage

1. Select an input video file with AceNodes video loader
2. Load an audio file using ComfyUI audio loader
3. (Optional) Set a seed value for reproducible results
4. Connect to the LatentSync node
5. Run the workflow

The processed video will be saved in ComfyUI's output directory.

### Node Parameters:
- `video_path`: Path to input video file
- `audio`: Audio input from AceNodes audio loader
- `seed`: Random seed for reproducible results (default: 1247)


## Note on Model Downloads

On first use, the node will automatically download required model files from HuggingFace:
- LatentSync UNet model
- Whisper model for audio processing

## Known Limitations

- Works best with clear, frontal face videos
- Currently does not support anime/cartoon faces
- Video should be at 25 FPS (will be automatically converted)
- Face should be visible throughout the video

## Credits

This is an unofficial implementation based on:
- [LatentSync](https://github.com/bytedance/LatentSync) by ByteDance Research
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI_AceNodes](https://github.com/hay86/ComfyUI_AceNodes)

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
