import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import os
import time
from typing import Dict, List, Tuple, Optional

from .s3_utils import get_s3_client
from .logger import logger


class S3FileList:
    """Helper class to manage S3 file listings with caching, similar to ComfyUI's pattern"""

    _cache: Dict[str, Tuple[List[str], float]] = {}

    @classmethod
    def get_files(cls, bucket: str, prefix: str = "") -> List[str]:
        """
        Get list of files from S3 with caching, similar to ComfyUI's get_filename_list
        """
        cache_key = f"{bucket}:{prefix}"

        # Check cache first
        if cache_key in cls._cache:
            cached_files, cached_time = cls._cache[cache_key]
            # Cache for 5 seconds
            if time.time() - cached_time < 5:
                return cached_files

        try:
            s3_client = get_s3_client()
            if not s3_client:
                return ["none"]

            response = s3_client.list_objects_v2(
                Bucket=bucket, Prefix=prefix if prefix else ""
            )

            if "Contents" not in response:
                return ["none"]

            # Filter for image files
            image_extensions = (".png", ".jpg", ".jpeg", ".webp")
            files = [
                obj["Key"].replace(prefix, "", 1) if prefix else obj["Key"]
                for obj in response["Contents"]
                if obj["Key"].lower().endswith(image_extensions)
            ]

            # Update cache
            cls._cache[cache_key] = (sorted(files) if files else ["none"], time.time())
            return sorted(files) if files else ["none"]

        except Exception as e:
            logger.error(f"Failed to list S3 files: {e}")
            return ["none"]


class LoadImageS3Interactive:
    current_bucket: Optional[str] = None
    current_prefix: Optional[str] = None

    @classmethod
    def INPUT_TYPES(s):
        """
        Define inputs following ComfyUI's pattern for dynamic dropdowns
        """
        files = ["none"]

        # If we have current bucket/prefix, try to get the file list
        if s.current_bucket:
            files = S3FileList.get_files(s.current_bucket, s.current_prefix or "")

        return {
            "required": {
                "bucket": ("STRING", {"default": s.current_bucket or ""}),
                "prefix": ("STRING", {"default": s.current_prefix or ""}),
                "image": (files,),  # Simple tuple format like ComfyUI uses
            }
        }

    CATEGORY = "image/input"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, bucket: str, prefix: str, image: str):
        """Load and process the selected image from S3"""
        try:
            # Update current values
            self.__class__.current_bucket = bucket
            self.__class__.current_prefix = prefix

            if image == "none":
                raise ValueError("No image selected")

            s3_client = get_s3_client()
            if not s3_client:
                raise ValueError("S3 client not initialized")

            # Construct the full S3 path
            s3_path = os.path.join(prefix if prefix else "", image)
            local_path = os.path.join("input", image)

            # Download the file
            s3_client.download_file(bucket, s3_path, local_path)

            try:
                img = Image.open(local_path)
                output_images = []
                output_masks = []

                for i in ImageSequence.Iterator(img):
                    i = ImageOps.exif_transpose(i)
                    if i.mode == "I":
                        i = i.point(lambda i: i * (1 / 255))
                    image = i.convert("RGB")
                    image = np.array(image).astype(np.float32) / 255.0
                    image = torch.from_numpy(image)[None,]
                    if "A" in i.getbands():
                        mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                        mask = 1.0 - torch.from_numpy(mask)
                    else:
                        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
                    output_images.append(image)
                    output_masks.append(mask.unsqueeze(0))

                if len(output_images) > 1:
                    output_image = torch.cat(output_images, dim=0)
                    output_mask = torch.cat(output_masks, dim=0)
                else:
                    output_image = output_images[0]
                    output_mask = output_masks[0]

                return (output_image, output_mask)

            finally:
                if os.path.exists(local_path):
                    os.remove(local_path)

        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise

    @classmethod
    def IS_CHANGED(s, bucket: str, prefix: str, image: str) -> bool:
        """
        Tell ComfyUI when to refresh the node.
        Following ComfyUI's pattern of using simple change detection.
        """
        return bucket != s.current_bucket or prefix != s.current_prefix
