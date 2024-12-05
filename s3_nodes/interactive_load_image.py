import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import os

from .s3_utils import get_s3_client
from .logger import logger


class LoadImageS3Interactive:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bucket": ("STRING", {"default": ""}),
                "prefix": ("STRING", {"default": ""}),
                "refresh": (["no", "yes"], {"default": "no"}),  # Add a refresh trigger
            }
        }

    CATEGORY = "image/input"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, bucket, prefix, refresh):
        """Load image from the specified bucket and prefix"""
        try:
            if not bucket:
                raise ValueError("Bucket name is required")

            # Get S3 client
            s3_client = get_s3_client()

            # List available images
            response = s3_client.list_objects_v2(
                Bucket=bucket, Prefix=prefix if prefix else ""
            )

            # Filter for images
            image_extensions = (".png", ".jpg", ".jpeg", ".webp")
            files = []
            if "Contents" in response:
                files = [
                    obj["Key"]
                    for obj in response["Contents"]
                    if obj["Key"].lower().endswith(image_extensions)
                ]

            if not files:
                raise ValueError(
                    f"No images found in bucket '{bucket}' with prefix '{prefix}'"
                )

            # For this example, let's just take the first image
            image_key = files[0]
            local_path = os.path.join("input", os.path.basename(image_key))

            # Download and process the image
            s3_client.download_file(bucket, image_key, local_path)

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
    def VALIDATE_INPUTS(s, bucket, prefix, refresh):
        try:
            if not bucket:
                return "Bucket name is required"

            # Try listing files to validate the connection
            s3_client = get_s3_client()
            s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix if prefix else "")

            return True

        except Exception as e:
            return str(e)
