import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import tempfile
import os
from .s3_utils import get_s3_client, list_s3_images
from .logger import logger


class LoadImageS3Interactive:
    def __init__(self):
        self.current_bucket = ""
        self.current_prefix = ""
        self.cached_images = []

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bucket": ("STRING", {"default": ""}),
                "prefix": ("STRING", {"default": ""}),
                # We'll set this to update dynamically based on bucket/prefix
                "image": ([""], {"default": ""}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"  # Used to force refresh of the dropdown
            },
        }

    CATEGORY = "image/input"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    @classmethod
    def update_image_list(s, bucket, prefix):
        """
        Get the list of available images for the given bucket and prefix.
        This is used to populate the dropdown options.
        """
        try:
            if not bucket:
                return [""]

            images = list_s3_images(bucket, prefix)
            if not images:
                logger.warning(
                    f"No images found in bucket '{bucket}' with prefix '{prefix}'"
                )
                return [""]

            return images

        except Exception as e:
            logger.error(f"Failed to update image list: {e}")
            return [""]

    # This special method is called by ComfyUI to get updated input types
    def CUSTOM_INPUTS(self, bucket, prefix, **kwargs):
        """
        Update the available options for the image dropdown whenever bucket or prefix changes.
        """
        # Only update if bucket or prefix has changed
        if bucket != self.current_bucket or prefix != self.current_prefix:
            self.current_bucket = bucket
            self.current_prefix = prefix
            self.cached_images = self.update_image_list(bucket, prefix)

        return {
            "required": {
                "bucket": ("STRING", {"default": bucket}),
                "prefix": ("STRING", {"default": prefix}),
                "image": (
                    self.cached_images,
                    {"default": self.cached_images[0] if self.cached_images else ""},
                ),
            }
        }

    def load_image(self, bucket, prefix, image, unique_id=None):
        """
        Load an image from S3 and return it as a tensor.

        Args:
            bucket (str): S3 bucket name
            prefix (str): Path prefix in the bucket
            image (str): Selected image key
            unique_id: Ignored, used only for refresh

        Returns:
            tuple: (image_tensor, mask_tensor)
        """
        try:
            s3_client = get_s3_client()

            # Create a temporary file to store the downloaded image
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                try:
                    # Download the file from S3
                    s3_client.download_file(bucket, image, temp_file.name)

                    # Process the image
                    img = Image.open(temp_file.name)
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
                            mask = (
                                np.array(i.getchannel("A")).astype(np.float32) / 255.0
                            )
                            mask = 1.0 - torch.from_numpy(mask)
                        else:
                            mask = torch.zeros(
                                (64, 64), dtype=torch.float32, device="cpu"
                            )

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
                    # Clean up the temporary file
                    if os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)

        except Exception as e:
            logger.error(f"Failed to load image from S3: {e}")
            raise

    @classmethod
    def IS_CHANGED(s, bucket, prefix, image, unique_id):
        """
        Check if the available images have changed.
        Forces a refresh of the dropdown by always returning True.
        """
        return True

    @classmethod
    def VALIDATE_INPUTS(s, bucket, prefix, image, unique_id):
        """
        Validate the inputs and update the available images list.
        """
        try:
            if not bucket:
                return "Bucket name is required"

            return True

        except Exception as e:
            return str(e)
