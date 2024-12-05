import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import os

from .s3_utils import get_s3_client
from .logger import logger


class LoadImageS3Interactive:
    s3_client = None
    current_bucket = None
    current_prefix = None

    @classmethod
    def get_s3_client(cls):
        """
        Get or create S3 client using lazy initialization
        """
        if cls.s3_client is None:
            try:
                cls.s3_client = get_s3_client()
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {e}")
                return None
        return cls.s3_client

    @classmethod
    def list_s3_images(cls, bucket, prefix=""):
        """
        List images in the S3 bucket/prefix
        Returns at least ["none"] to ensure the dropdown always has a valid option
        """
        try:
            if not bucket:
                return ["none"]

            s3_client = cls.get_s3_client()
            if s3_client is None:
                return ["none"]

            response = s3_client.list_objects_v2(
                Bucket=bucket, Prefix=prefix if prefix else ""
            )

            if "Contents" not in response:
                return ["none"]

            # Filter for image files
            image_extensions = (".png", ".jpg", ".jpeg", ".webp")
            files = [
                obj["Key"]
                for obj in response["Contents"]
                if obj["Key"].lower().endswith(image_extensions)
            ]

            # Remove prefix from filenames if needed
            if prefix:
                files = [f.replace(prefix, "", 1) for f in files]

            return sorted(files) if files else ["none"]

        except Exception as e:
            logger.error(f"Failed to list S3 files: {e}")
            return ["none"]

    @classmethod
    def INPUT_TYPES(s):
        """
        Define the input types for the node
        Always provide a valid list for the dropdown, even if empty
        """
        try:
            # Get the current files list if we have a bucket and prefix
            if s.current_bucket:
                files = s.list_s3_images(s.current_bucket, s.current_prefix or "")
            else:
                files = ["none"]

            return {
                "required": {
                    "bucket": (
                        "STRING",
                        {"default": s.current_bucket or "", "multiline": False},
                    ),
                    "prefix": (
                        "STRING",
                        {"default": s.current_prefix or "", "multiline": False},
                    ),
                    "image": (
                        files,
                    ),  # ComfyUI expects a tuple with the list as first element
                }
            }
        except Exception as e:
            logger.error(f"Error in INPUT_TYPES: {e}")
            return {
                "required": {
                    "bucket": ("STRING", {"default": ""}),
                    "prefix": ("STRING", {"default": ""}),
                    "image": (["none"],),  # Always provide a valid default
                }
            }

    CATEGORY = "image/input"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, bucket, prefix, image):
        """
        Load and process the selected image from S3
        """
        try:
            # Update the class's stored bucket/prefix
            self.__class__.current_bucket = bucket
            self.__class__.current_prefix = prefix

            # Don't try to load if we got the "none" placeholder
            if image == "none":
                raise ValueError("No image selected")

            # Get S3 client
            s3_client = self.get_s3_client()
            if s3_client is None:
                raise ValueError("S3 client not initialized")

            # Construct the full S3 path
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
    def IS_CHANGED(s, bucket, prefix, image):
        """
        Tell ComfyUI when to refresh the node
        """
        return bucket != s.current_bucket or prefix != s.current_prefix
