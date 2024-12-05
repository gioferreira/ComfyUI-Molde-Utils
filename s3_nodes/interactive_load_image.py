import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import os

from .s3_utils import get_s3_client
from .logger import logger

# Create a singleton S3 client instance
S3_CLIENT = get_s3_client()


class LoadImageS3Interactive:
    @classmethod
    def INPUT_TYPES(s):
        # First, define the bucket and prefix inputs
        bucket = ""  # Default empty bucket
        prefix = ""  # Default empty prefix

        try:
            # Try to list files from S3 if bucket is provided
            if bucket:
                response = S3_CLIENT.list_objects_v2(
                    Bucket=bucket, Prefix=prefix if prefix else ""
                )

                if "Contents" in response:
                    # Filter for image files
                    image_extensions = (".png", ".jpg", ".jpeg", ".webp")
                    files = [
                        obj["Key"]
                        for obj in response["Contents"]
                        if obj["Key"].lower().endswith(image_extensions)
                    ]

                    if prefix:
                        files = [f.replace(prefix, "", 1) for f in files]
                else:
                    files = []
            else:
                files = []

        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            files = []

        # Return the input configuration with all three inputs
        return {
            "required": {
                "bucket": ("STRING", {"default": ""}),
                "prefix": ("STRING", {"default": ""}),
                "image": (sorted(files), {"default": "", "image_upload": False}),
            }
        }

    CATEGORY = "image/input"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, bucket, prefix, image):
        """
        Load the selected image from S3.
        This follows the original implementation's pattern but uses our bucket/prefix.
        """
        try:
            # Construct the full S3 path
            s3_path = os.path.join(prefix if prefix else "", image)

            # Create a local path for temporary storage
            local_path = os.path.join("input", image)

            # Download the file
            S3_CLIENT.download_file(bucket, s3_path, local_path)

            # Process the image using the original implementation's code
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

            # Clean up the local file
            if os.path.exists(local_path):
                os.remove(local_path)

            if len(output_images) > 1:
                output_image = torch.cat(output_images, dim=0)
                output_mask = torch.cat(output_masks, dim=0)
            else:
                output_image = output_images[0]
                output_mask = output_masks[0]

            return (output_image, output_mask)

        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise

    @classmethod
    def IS_CHANGED(s, bucket, prefix, image):
        """
        Tell ComfyUI to always check for updates
        """
        return True
