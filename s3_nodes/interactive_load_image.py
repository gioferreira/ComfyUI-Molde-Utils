import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import os

from .s3_utils import get_s3_client
from .logger import logger


class LoadImageS3Interactive:
    # Class-level variables to track state
    _current_bucket = ""
    _current_prefix = ""
    _current_files = [""]

    @classmethod
    def INPUT_TYPES(s):
        """
        Define inputs. If we have current bucket/prefix, show files from there.
        """
        if s._current_bucket and s._current_prefix:
            try:
                s3_client = get_s3_client()
                response = s3_client.list_objects_v2(
                    Bucket=s._current_bucket,
                    Prefix=s._current_prefix if s._current_prefix else "",
                )

                if "Contents" in response:
                    image_extensions = (".png", ".jpg", ".jpeg", ".webp")
                    s._current_files = [
                        obj["Key"].replace(s._current_prefix, "", 1)
                        if s._current_prefix
                        else obj["Key"]
                        for obj in response["Contents"]
                        if obj["Key"].lower().endswith(image_extensions)
                    ]
                    if not s._current_files:
                        s._current_files = [""]
                else:
                    s._current_files = [""]
            except Exception as e:
                logger.error(f"Failed to list S3 files: {e}")
                s._current_files = [""]

        return {
            "required": {
                "bucket": ("STRING", {"default": s._current_bucket}),
                "prefix": ("STRING", {"default": s._current_prefix}),
                "image": (sorted(s._current_files),),
            }
        }

    CATEGORY = "image/input"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, bucket, prefix, image):
        """Load and process the selected image from S3"""
        try:
            # Update class state
            self.__class__._current_bucket = bucket
            self.__class__._current_prefix = prefix

            if not bucket:
                raise ValueError("Bucket name is required")

            if image == "":
                raise ValueError("No image selected")

            s3_client = get_s3_client()
            s3_path = os.path.join(prefix if prefix else "", image)
            local_path = os.path.join("input", image)

            # Download the file
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
        Tell ComfyUI to consider the node changed and force refresh when bucket/prefix change
        """
        changed = bucket != s._current_bucket or prefix != s._current_prefix
        if changed:
            # This will cause ComfyUI to call INPUT_TYPES again
            s._current_bucket = bucket
            s._current_prefix = prefix
        return changed

    @classmethod
    def VALIDATE_INPUTS(s, bucket, prefix, image):
        """Validate the inputs"""
        try:
            if not bucket:
                return "Bucket name is required"

            if image == "":
                return "No image selected"

            return True

        except Exception as e:
            return str(e)
