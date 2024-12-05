import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import os

from .s3_utils import get_s3_client
from .logger import logger


class LoadImageS3Interactive:
    @classmethod
    def INPUT_TYPES(s):
        """
        Called by ComfyUI when:
        1. The node is first created
        2. The graph is refreshed
        3. getNodeDefs is called
        """
        return {
            "required": {
                "bucket": ("STRING", {"default": ""}),
                "prefix": ("STRING", {"default": ""}),
                # For the image dropdown, ComfyUI will use the first element of this tuple
                # during refresh operations
                "image": ([""],),
            }
        }

    CATEGORY = "image/input"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def get_images_for_bucket(self, bucket, prefix):
        """Get list of images from S3"""
        try:
            if not bucket:
                return [""]

            s3_client = get_s3_client()
            response = s3_client.list_objects_v2(
                Bucket=bucket, Prefix=prefix if prefix else ""
            )

            if "Contents" not in response:
                return [""]

            # Filter for image files
            image_extensions = (".png", ".jpg", ".jpeg", ".webp")
            files = [
                obj["Key"].replace(prefix, "", 1) if prefix else obj["Key"]
                for obj in response["Contents"]
                if obj["Key"].lower().endswith(image_extensions)
            ]

            return sorted(files) if files else [""]

        except Exception as e:
            logger.error(f"Failed to list S3 files: {e}")
            return [""]

    def load_image(self, bucket, prefix, image):
        """Load and process the selected image"""
        try:
            if not bucket:
                raise ValueError("Bucket name is required")

            if image == "":
                # Get fresh list of images
                images = self.get_images_for_bucket(bucket, prefix)
                if not images or images[0] == "":
                    raise ValueError(
                        f"No images found in bucket '{bucket}' with prefix '{prefix}'"
                    )
                image = images[0]

            s3_client = get_s3_client()
            s3_path = os.path.join(prefix if prefix else "", image)
            local_path = os.path.join("input", image)

            # Download and process the image
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
    def VALIDATE_INPUTS(s, bucket, prefix, image):
        """Validate the inputs"""
        try:
            if not bucket:
                return "Bucket name is required"
            return True
        except Exception as e:
            return str(e)
