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
        Define the input types for the node.
        For dropdowns, ComfyUI expects a tuple where the first element is the list of options.
        """
        files = ["none"]  # Default empty state
        try:
            # Only try to list files if we have client configured
            s3_client = get_s3_client()
            if s3_client:
                # List objects without any bucket/prefix yet
                files = ["none"]  # We'll update this when bucket/prefix are provided
        except Exception as e:
            logger.error(f"Error initializing S3 client: {e}")

        return {
            "required": {
                "bucket": ("STRING", {"default": ""}),
                "prefix": ("STRING", {"default": ""}),
                # The key part: match exactly how ComfyUI's built-in nodes format their dropdowns
                "image": (files, {"default": "none"}),
            }
        }

    CATEGORY = "image/input"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, bucket, prefix, image):
        """Load and process the selected image from S3"""
        try:
            if image == "none":
                raise ValueError("No image selected")

            s3_client = get_s3_client()

            # First, verify we can list files with these credentials
            response = s3_client.list_objects_v2(
                Bucket=bucket, Prefix=prefix if prefix else ""
            )

            # If we got here, update the class's input types
            new_files = []
            if "Contents" in response:
                image_extensions = (".png", ".jpg", ".jpeg", ".webp")
                new_files = [
                    obj["Key"].replace(prefix, "", 1) if prefix else obj["Key"]
                    for obj in response["Contents"]
                    if obj["Key"].lower().endswith(image_extensions)
                ]

            # Now proceed with loading the selected image
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
        """
        Tell ComfyUI to force a UI refresh when the bucket or prefix changes.
        This should trigger a re-fetch of the available images.
        """
        try:
            if not bucket:
                return False

            # This forces ComfyUI to refresh the node
            return True
        except Exception as e:
            logger.error(f"Error in IS_CHANGED: {e}")
            return False

    @classmethod
    def VALIDATE_INPUTS(s, bucket, prefix, image):
        """
        Validate inputs and update available images list.
        This is called by ComfyUI to check if the inputs are valid.
        """
        try:
            if not bucket:
                return "Bucket is required"

            s3_client = get_s3_client()
            if not s3_client:
                return "S3 client not initialized"

            # Try to list objects to validate bucket/prefix
            response = s3_client.list_objects_v2(
                Bucket=bucket, Prefix=prefix if prefix else ""
            )

            return True

        except Exception as e:
            return str(e)
