import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import os

from .s3_utils import get_s3_client
from .logger import logger


class LoadImageS3Interactive:
    # Class variables to track state
    _current_bucket = ""
    _current_prefix = ""
    _current_images = [""]

    @classmethod
    def INPUT_TYPES(s):
        """
        This method is called by ComfyUI whenever it needs to know what inputs the node accepts.
        """
        # First, define the inputs including our image dropdown
        return {
            "required": {
                "bucket": ("STRING", {"default": s._current_bucket}),
                "prefix": ("STRING", {"default": s._current_prefix}),
                "image": (s._current_images,),  # Use our cached list of images
                "refresh": (
                    ["no", "yes"],
                    {"default": "no"},
                ),  # Keep refresh trigger for manual updates
            }
        }

    CATEGORY = "image/input"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    @classmethod
    def update_image_list(cls, bucket, prefix):
        """
        Update the list of available images for a given bucket and prefix.
        Returns the list of images and updates the class state.
        """
        try:
            if not bucket:
                cls._current_images = [""]
                return cls._current_images

            s3_client = get_s3_client()
            response = s3_client.list_objects_v2(
                Bucket=bucket, Prefix=prefix if prefix else ""
            )

            image_extensions = (".png", ".jpg", ".jpeg", ".webp")
            files = []
            if "Contents" in response:
                files = [
                    obj["Key"].replace(prefix, "", 1) if prefix else obj["Key"]
                    for obj in response["Contents"]
                    if obj["Key"].lower().endswith(image_extensions)
                ]

            # Update our cached list
            cls._current_images = sorted(files) if files else [""]
            cls._current_bucket = bucket
            cls._current_prefix = prefix

            return cls._current_images

        except Exception as e:
            logger.error(f"Failed to list S3 files: {e}")
            cls._current_images = [""]
            return cls._current_images

    def load_image(self, bucket, prefix, image, refresh):
        """Load and process the selected image"""
        try:
            if not bucket:
                raise ValueError("Bucket name is required")

            # If either bucket or prefix changed, or refresh is requested,
            # update our image list
            if (
                bucket != self._current_bucket
                or prefix != self._current_prefix
                or refresh == "yes"
            ):
                self.update_image_list(bucket, prefix)

            if not image or image == "":
                raise ValueError(
                    f"No images found in bucket '{bucket}' with prefix '{prefix}'"
                )

            # Get S3 client and construct paths
            s3_client = get_s3_client()
            s3_path = os.path.join(prefix if prefix else "", image)
            local_path = os.path.join("input", os.path.basename(image))

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
    def IS_CHANGED(s, bucket, prefix, image, refresh):
        """Tell ComfyUI when to refresh the node"""
        # Refresh when bucket or prefix changes, or when refresh is requested
        changed = (
            bucket != s._current_bucket
            or prefix != s._current_prefix
            or refresh == "yes"
        )

        if changed:
            # Update our image list
            s.update_image_list(bucket, prefix)
        return changed

    @classmethod
    def VALIDATE_INPUTS(s, bucket, prefix, image, refresh):
        """Validate the inputs"""
        try:
            if not bucket:
                return "Bucket name is required"

            if image == "":
                return "No image selected"

            return True

        except Exception as e:
            return str(e)
