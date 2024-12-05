import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import tempfile
import os
from .s3_utils import get_s3_client, list_s3_images
from .logger import logger


class LoadImageS3Interactive:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bucket": ("STRING", {"default": ""}),
                "prefix": ("STRING", {"default": ""}),
                "image": ("STRING", {"default": ""}),  # Will be populated with files
            }
        }

    CATEGORY = "image/input"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, bucket, prefix, image):
        """
        Load an image from S3 and return it as a tensor.
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
    def IS_CHANGED(s, bucket, prefix, image):
        """
        Tell ComfyUI to update when any input changes
        """
        return True

    @classmethod
    def VALIDATE_INPUTS(s, bucket, prefix, image):
        """
        Validate the inputs and update the available images list.
        """
        try:
            if not bucket:
                return "Bucket name is required"

            return True

        except Exception as e:
            return str(e)

    # This method gets called when inputs change
    def update(self, bucket, prefix, **kwargs):
        """
        Update available images based on bucket and prefix
        """
        if not bucket:
            return {"image": [""]}

        try:
            files = list_s3_images(bucket, prefix)
            return {"image": sorted(files) if files else [""]}
        except Exception as e:
            logger.error(f"Failed to list images: {e}")
            return {"image": [""]}
