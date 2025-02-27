# import os
# import tempfile
# import zipfile
import numpy as np
import torch
from PIL import Image  # , ImageOps
from .logger import logger
from comfy.utils import common_upscale


class ProcessZipImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "resize_mode": (
                    ["disabled", "resize", "crop", "pad"],
                    {"default": "disabled"},
                ),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "crop_position": (
                    ["center", "top", "bottom", "left", "right"],
                    {"default": "center"},
                ),
                "upscale_method": (
                    ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
                    {"default": "lanczos"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_images"
    CATEGORY = "image/processing"

    def process_images(
        self,
        images,
        resize_mode="disabled",
        width=512,
        height=512,
        crop_position="center",
        upscale_method="lanczos",
    ):
        """Process batch of images with various transformations"""
        try:
            # If no resizing is needed, return the original images
            if resize_mode == "disabled":
                return (images,)

            batch_size, img_height, img_width, channels = images.shape

            # Process each image
            processed_images = []

            for i in range(batch_size):
                img_tensor = images[i]

                if resize_mode == "resize":
                    # Simple resize (might change aspect ratio)
                    img_resized = img_tensor.movedim(-1, 0).unsqueeze(
                        0
                    )  # [C, H, W] -> [1, C, H, W]
                    img_resized = common_upscale(
                        img_resized, width, height, upscale_method, "center"
                    )
                    img_resized = img_resized.squeeze(0).movedim(
                        0, -1
                    )  # [1, C, H, W] -> [H, W, C]
                    processed_images.append(img_resized)

                elif resize_mode == "crop":
                    # Resize and crop to target dimensions
                    img_np = img_tensor.numpy()
                    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))

                    # Calculate target aspect ratio
                    target_ratio = width / height
                    img_ratio = img_width / img_height

                    if img_ratio > target_ratio:  # Image is wider
                        # Resize height to match target
                        new_height = height
                        new_width = int(img_ratio * new_height)
                        img_pil = img_pil.resize(
                            (new_width, new_height),
                            getattr(Image, "Resampling", Image).LANCZOS,
                        )

                        # Crop width
                        if crop_position == "center":
                            left = (new_width - width) // 2
                        elif crop_position == "left":
                            left = 0
                        elif crop_position == "right":
                            left = new_width - width
                        else:
                            left = (new_width - width) // 2

                        img_pil = img_pil.crop((left, 0, left + width, height))

                    else:  # Image is taller
                        # Resize width to match target
                        new_width = width
                        new_height = int(new_width / img_ratio)
                        img_pil = img_pil.resize(
                            (new_width, new_height),
                            getattr(Image, "Resampling", Image).LANCZOS,
                        )

                        # Crop height
                        if crop_position == "center":
                            top = (new_height - height) // 2
                        elif crop_position == "top":
                            top = 0
                        elif crop_position == "bottom":
                            top = new_height - height
                        else:
                            top = (new_height - height) // 2

                        img_pil = img_pil.crop((0, top, width, top + height))

                    # Convert back to tensor
                    img_np = np.array(img_pil).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_np)
                    processed_images.append(img_tensor)

                elif resize_mode == "pad":
                    # Resize and pad to maintain aspect ratio
                    img_np = img_tensor.numpy()
                    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))

                    # Calculate target aspect ratio
                    target_ratio = width / height
                    img_ratio = img_width / img_height

                    if img_ratio > target_ratio:  # Image is wider
                        # Resize width to match target
                        new_width = width
                        new_height = int(new_width / img_ratio)
                        resized_img = img_pil.resize(
                            (new_width, new_height),
                            getattr(Image, "Resampling", Image).LANCZOS,
                        )

                        # Create padded image
                        padded_img = Image.new("RGB", (width, height), (0, 0, 0))
                        paste_y = (height - new_height) // 2
                        padded_img.paste(resized_img, (0, paste_y))

                    else:  # Image is taller
                        # Resize height to match target
                        new_height = height
                        new_width = int(img_ratio * new_height)
                        resized_img = img_pil.resize(
                            (new_width, new_height),
                            getattr(Image, "Resampling", Image).LANCZOS,
                        )

                        # Create padded image
                        padded_img = Image.new("RGB", (width, height), (0, 0, 0))
                        paste_x = (width - new_width) // 2
                        padded_img.paste(resized_img, (paste_x, 0))

                    # Convert back to tensor
                    img_np = np.array(padded_img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_np)
                    processed_images.append(img_tensor)

            # Stack processed images into a batch
            if processed_images:
                return (torch.stack(processed_images),)
            else:
                return (images,)

        except Exception as e:
            logger.error(f"Error processing images: {e}")
            raise
