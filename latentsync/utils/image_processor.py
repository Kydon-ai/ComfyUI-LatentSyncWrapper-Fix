# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from latentsync.utils.util import read_video, write_video
from torchvision import transforms
import cv2
from einops import rearrange
import torch
import numpy as np
from typing import Union
from .affine_transform import AlignRestore
from .face_detector import FaceDetector


def load_fixed_mask(resolution: int, mask_image_path="latentsync/utils/mask.png") -> torch.Tensor:
    mask_image = cv2.imread(mask_image_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.resize(mask_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4) / 255.0
    mask_image = rearrange(torch.from_numpy(mask_image), "h w c -> c h w")
    return mask_image


class ImageProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu", mask_image=None):
        self.resolution = resolution
        self.resize = transforms.Resize(
            (resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)

        self.restorer = AlignRestore(resolution=resolution, device=device)

        if mask_image is None:
            self.mask_image = load_fixed_mask(resolution)
        else:
            self.mask_image = mask_image

        if device == "cpu":
            self.face_detector = None
        else:
            self.face_detector = FaceDetector(device=device)
    # NEW:新增函数，处理面部
    def detect_facial_landmarks(self, image: np.ndarray):  
        height, width, _ = image.shape  
        results = self.face_mesh.process(image)  
        if not results.multi_face_landmarks:  # Face not detected  
            print("Skipping frame: No face detected")  
            return None  # Return None instead of raising an error  
        face_landmarks = results.multi_face_landmarks[0]  # Only use the first face in the image  
        landmark_coordinates = [  
            (int(landmark.x * width), int(landmark.y * height)) for landmark in face_landmarks.landmark  
        ]  # x means width, y means height  
        return landmark_coordinates
    
    def affine_transform(self, image: torch.Tensor) -> np.ndarray:
        if self.face_detector is None:
            raise NotImplementedError("Using the CPU for face detection is not supported")
        bbox, landmark_2d_106 = self.face_detector(image)
        if bbox is None:
            # raise RuntimeError("Face not detected")
            print("[LatentSyncWrapper Fix]: Face not detected")
            # mock人脸数据,防止数据校验出错。更好的解决方法，应该是替换校验逻辑为先检测bbox标记再校验。这只是快速解决方法但并不完美！
            landmark_2d_106 = np.array([
                [388, 639], [205, 317], [241, 542], [254, 564], [270, 583], [288, 601], [309, 616], [332, 629],
                [358, 637], [203, 343], [204, 368], [206, 393], [209, 418], [213, 443], [218, 469], [223, 494],
                [231, 519], [558, 311], [528, 535], [516, 557], [501, 577], [484, 595], [464, 612], [442, 626],
                [417, 635], [560, 336], [560, 362], [559, 387], [556, 412], [553, 437], [549, 462], [544, 487],
                [537, 511], [302, 350], [304, 334], [270, 338], [284, 346], [322, 347], [304, 334], [339, 341],
                [304, 321], [284, 326], [324, 327], [237, 302], [262, 292], [289, 287], [347, 298], [319, 290],
                [257, 275], [288, 265], [350, 282], [322, 269], [322, 521], [387, 553], [353, 525], [338, 537],
                [359, 549], [420, 523], [436, 534], [415, 547], [386, 527], [452, 517], [386, 514], [369, 495],
                [343, 505], [332, 521], [354, 515], [401, 494], [429, 502], [442, 517], [419, 513], [385, 498],
                [381, 330], [382, 363], [382, 395], [356, 343], [346, 412], [334, 441], [349, 451], [365, 455],
                [383, 460], [407, 342], [420, 411], [432, 440], [418, 451], [402, 455], [382, 428], [461, 347],
                [460, 332], [425, 341], [442, 345], [480, 343], [460, 332], [494, 335], [459, 319], [439, 325],
                [479, 323], [414, 298], [442, 290], [471, 288], [498, 292], [525, 300], [412, 283], [439, 271],
                [472, 267], [504, 276]
            ])
            bbox = False
        else:
            print("[LatentSyncWrapper Fix]: Face detected!")
            # 打印合适的landmark_2d_106的所有元素，从这里取出某一个合理的landmark_2d_106进行mock
            # np.set_printoptions(threshold=np.inf)
            # print(landmark_2d_106)
            bbox = True
            
        pt_left_eye = np.mean(landmark_2d_106[[43, 48, 49, 51, 50]], axis=0)  # left eyebrow center
        pt_right_eye = np.mean(landmark_2d_106[101:106], axis=0)  # right eyebrow center
        pt_nose = np.mean(landmark_2d_106[[74, 77, 83, 86]], axis=0)  # nose center

        landmarks3 = np.round([pt_left_eye, pt_right_eye, pt_nose])

        face, affine_matrix  = self.restorer.align_warp_face(image.copy(), landmarks3=landmarks3, smooth=True)
        box = [0, 0, face.shape[1], face.shape[0]]  # x1, y1, x2, y2
        face = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
        face = rearrange(torch.from_numpy(face), "h w c -> c h w")
        return bbox, face, box, affine_matrix

    def preprocess_fixed_mask_image(self, image: torch.Tensor, affine_transform=False):
        if affine_transform:
            bbox, image, _, _ = self.affine_transform(image)
        else:
            image = self.resize(image)
        pixel_values = self.normalize(image / 255.0)
        masked_pixel_values = pixel_values * self.mask_image
        return pixel_values, masked_pixel_values, self.mask_image[0:1]

    def prepare_masks_and_masked_images(self, images: Union[torch.Tensor, np.ndarray], affine_transform=False):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")

        results = [self.preprocess_fixed_mask_image(image, affine_transform=affine_transform) for image in images]

        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)

    def process_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")
        images = self.resize(images)
        pixel_values = self.normalize(images / 255.0)
        return pixel_values


class VideoProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu"):
        self.image_processor = ImageProcessor(resolution, device)

    def affine_transform_video(self, video_path):
        video_frames = read_video(video_path, change_fps=False)
        results = []
        for frame in video_frames:
            bbox, frame, _, _ = self.image_processor.affine_transform(frame)
            results.append(frame)
        results = torch.stack(results)

        results = rearrange(results, "f c h w -> f h w c").numpy()
        return results


if __name__ == "__main__":
    video_processor = VideoProcessor(256, "cuda")
    video_frames = video_processor.affine_transform_video("assets/demo2_video.mp4")
    write_video("output.mp4", video_frames, fps=25)
