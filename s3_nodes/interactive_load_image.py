import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import os

from .s3_utils import get_s3_client
from .logger import logger


class LoadImageS3Interactive:
    current_bucket = None
    current_prefix = None
    s3_client = None

    @classmethod
    def get_s3_client(cls):
        """
        Get or create S3 client
        """
        if cls.s3_client is None:
            cls.s3_client = get_s3_client()
        return cls.s3_client

    @classmethod
    def get_image_list(cls, bucket, prefix):
        """
        Get list of images from S3 bucket/prefix combination
        """
        try:
            if not bucket:
                return []

            s3_client = cls.get_s3_client()
            response = s3_client.list_objects_v2(
                Bucket=bucket, Prefix=prefix if prefix else ""
            )

            if "Contents" not in response:
                return []

            image_extensions = (".png", ".jpg", ".jpeg", ".webp")
            files = [
                obj["Key"]
                for obj in response["Contents"]
                if obj["Key"].lower().endswith(image_extensions)
            ]

            if prefix:
                files = [f.replace(prefix, "", 1) for f in files]

            return sorted(files) if files else []

        except Exception as e:
            logger.error(f"Failed to list S3 files: {e}")
            return []

    @classmethod
    def INPUT_TYPES(s):
        files = []
        if s.current_bucket:
            files = s.get_image_list(s.current_bucket, s.current_prefix or "")

        return {
            "required": {
                "bucket": ("STRING", {"default": s.current_bucket or ""}),
                "prefix": ("STRING", {"default": s.current_prefix or ""}),
                "image": (files, {"default": files[0] if files else ""}),
            }
        }

    CATEGORY = "image/input"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, bucket, prefix, image):
        try:
            self.__class__.current_bucket = bucket
            self.__class__.current_prefix = prefix

            s3_client = self.get_s3_client()
            s3_path = os.path.join(prefix if prefix else "", image)
            local_path = os.path.join("input", image)

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
    def IS_CHANGED(s, bucket, prefix, image):
        return bucket != s.current_bucket or prefix != s.current_prefix
