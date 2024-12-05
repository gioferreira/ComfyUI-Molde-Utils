import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import os

from .s3_utils import get_s3_client
from .logger import logger


class LoadImageS3Interactive:
    # Static configuration for the node
    s3_bucket = None
    s3_prefix = None

    @classmethod
    def setup_s3_connection(cls, bucket, prefix=""):
        """
        Set up the S3 connection that will be used by this node.
        This should be called before the node is used.
        """
        cls.s3_bucket = bucket
        cls.s3_prefix = prefix
        logger.info(f"S3 connection configured for bucket: {bucket}, prefix: {prefix}")

    @classmethod
    def INPUT_TYPES(s):
        """
        Get the available files from S3, similar to how LoadImage gets files from disk
        """
        try:
            if not s.s3_bucket:
                return {"required": {"image": (["Please configure S3 bucket first"],)}}

            # Get S3 client
            s3_client = get_s3_client()

            # List objects in bucket/prefix
            response = s3_client.list_objects_v2(
                Bucket=s.s3_bucket, Prefix=s.s3_prefix if s.s3_prefix else ""
            )

            # Filter for image files
            files = []
            if "Contents" in response:
                image_extensions = (".png", ".jpg", ".jpeg", ".webp")
                files = [
                    obj["Key"].replace(s.s3_prefix, "", 1)
                    if s.s3_prefix
                    else obj["Key"]
                    for obj in response["Contents"]
                    if obj["Key"].lower().endswith(image_extensions)
                ]

            if not files:
                files = ["No images found"]

            # Return the input configuration
            return {"required": {"image": (sorted(files),)}}

        except Exception as e:
            logger.error(f"Error listing S3 files: {e}")
            return {"required": {"image": (["Error listing S3 files"],)}}

    CATEGORY = "image/input"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image):
        """Load and process the selected image"""
        try:
            if not self.s3_bucket:
                raise ValueError("S3 bucket not configured")

            if image in [
                "No images found",
                "Error listing S3 files",
                "Please configure S3 bucket first",
            ]:
                raise ValueError("No valid image selected")

            # Get the S3 client
            s3_client = get_s3_client()

            # Construct the full S3 path
            s3_path = os.path.join(self.s3_prefix if self.s3_prefix else "", image)
            local_path = os.path.join("input", image)

            # Download the file
            s3_client.download_file(self.s3_bucket, s3_path, local_path)

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
    def update_s3_config(cls, bucket, prefix=""):
        """
        Update the S3 configuration. This should trigger a refresh.
        """
        if bucket != cls.s3_bucket or prefix != cls.s3_prefix:
            cls.setup_s3_connection(bucket, prefix)
            return True
        return False
