# ComfyUI-LatentSyncWrapper

Unofficial [LatentSync](https://github.com/bytedance/LatentSync) implementation for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) on windows.

This node provides lip-sync capabilities in ComfyUI using ByteDance's LatentSync model. It allows you to synchronize video lips with audio input.

![Screenshot 2025-01-02 210507](https://github.com/user-attachments/assets/df4c83a9-d170-4eb2-b406-38fb7a93c6aa)


https://github.com/user-attachments/assets/49c40cf4-5db1-46c5-99a4-7fbb2031c907



## Prerequisites

Before installing this node, you must install the following in order:

1. [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed and working

## Windows Installation Notes
1. Install FFmpeg:
   - Download from [FFmpeg Builds](https://github.com/BtbN/FFmpeg-Builds/releases)
   - Extract to a folder
   - Add the bin folder to system PATH
   - Restart ComfyUI

2. If you get PYTHONPATH errors:
   - Make sure Python is in your system PATH
   - Try running ComfyUI as administrator
     
## Installation

Only proceed with installation after confirming all prerequisites are installed and working.

1. Clone this repository into your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ShymuelRonen/ComfyUI-LatentSyncWrapper.git
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
## Note on Model Downloads

On first use, the node will automatically download required model files from HuggingFace:
- LatentSync UNet model
- Whisper model for audio processing
- You can also manualy download the models from HuggingFace repo: https://huggingface.co/chunyu-li/LatentSync, the checkpoints should appear as follows:

```
./checkpoints/
|-- latentsync_unet.pt
|-- whisper
|   `-- tiny.pt
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


## Known Limitations

- Works best with clear, frontal face videos
- Currently does not support anime/cartoon faces
- Video should be at 25 FPS (will be automatically converted)
- Face should be visible throughout the video

### NEW - Video Length Adjuster Node
A complementary node that helps manage video length and synchronization with audio.

#### Features:
- Displays video and audio duration information
- Three modes of operation:
  - `normal`: Passes through video frames with added padding to prevent frame loss
  - `pingpong`: Creates a forward-backward loop of the video sequence
  - `loop_to_audio`: Extends video by repeating frames to match audio duration

#### Usage:
1. Place the Video Length Adjuster between your video input and the LatentSync node
2. Connect audio to both the Video Length Adjuster and Video Combine nodes
3. Select desired mode based on your needs:
   - Use `normal` for standard lip-sync
   - Use `pingpong` for back-and-forth animation
   - Use `loop_to_audio` to match longer audio durations

#### Example Workflow:
1. Load Video (Upload) → Video frames output
2. Load Audio → Audio output
3. Connect both to Video Length Adjuster
4. Video Length Adjuster → LatentSync Node
5. LatentSync Node + Original Audio → Video Combine

## Credits

This is an unofficial implementation based on:
- [LatentSync](https://github.com/bytedance/LatentSync) by ByteDance Research
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
