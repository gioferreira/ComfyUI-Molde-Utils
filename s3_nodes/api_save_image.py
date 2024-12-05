import os
import uuid
import json
import tempfile
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .s3_utils import get_s3_client
from .logger import logger


class SaveImageS3API:
    def __init__(self):
        self.compress_level = 4  # Default compression level

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "bucket": ("STRING", {"default": ""}),
                "prefix": ("STRING", {"default": ""}),
                "compression": ("INT", {"default": 4, "min": 0, "max": 9, "step": 1}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("s3_uris",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "image/output"

    def save_images(
        self, images, bucket, prefix, compression=4, prompt=None, extra_pnginfo=None
    ):
        """
        Save images to S3 with UUID-based filenames.
        """
        try:
            s3_client = get_s3_client()
            s3_uris = []
            results = []

            # Prepare metadata if enabled
            metadata = None
            if prompt is not None or extra_pnginfo is not None:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            for image in images:
                # Convert the image tensor to a PIL Image
                i = 255.0 * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

                # Generate a unique filename using UUID
                filename = f"{uuid.uuid4()}.png"

                # Create the full S3 key (prefix + filename)
                s3_key = os.path.join(prefix, filename) if prefix else filename

                # Save the image to a temporary file first
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".png"
                ) as temp_file:
                    try:
                        # Save the image to the temporary file
                        img.save(
                            temp_file.name, pnginfo=metadata, compress_level=compression
                        )

                        # Upload to S3
                        s3_client.upload_file(temp_file.name, bucket, s3_key)

                        # Generate the S3 URI directly (replacing the removed utility function)
                        s3_uri = f"s3://{bucket}/{s3_key}"
                        s3_uris.append(s3_uri)

                        # Add to results for UI
                        results.append(
                            {
                                "filename": filename,
                                "subfolder": prefix,
                                "type": "output",
                            }
                        )

                    finally:
                        # Clean up the temporary file
                        if os.path.exists(temp_file.name):
                            os.unlink(temp_file.name)

            return {"ui": {"images": results}, "result": (s3_uris,)}

        except Exception as e:
            logger.error(f"Failed to save images to S3: {e}")
            raise

    @classmethod
    def VALIDATE_INPUTS(s, bucket, prefix, compression, **kwargs):
        """
        Validate the inputs.
        """
        try:
            if not bucket:
                return "Bucket name is required"

            if compression < 0 or compression > 9:
                return "Compression level must be between 0 and 9"

            return True

        except Exception as e:
            return str(e)
