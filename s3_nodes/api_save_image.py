import os
import json
import uuid
import base64
import tempfile
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from .s3_utils import get_s3_client
from .logger import logger


class SaveImageS3API:
    def __init__(self):
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "bucket": ("STRING", {"default": ""}),
                "prefix": ("STRING", {"default": ""}),
                "compression": ("INT", {"default": 4, "min": 0, "max": 9, "step": 1}),
                "include_workflow_metadata": (
                    ["enabled", "disabled"],
                    {"default": "enabled"},
                ),
            },
            "optional": {
                # Now accepts any string content
                "custom_metadata": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Custom metadata string or base64 content",
                    },
                ),
                # Switch to specify if the content is base64
                "metadata_is_base64": (["yes", "no"], {"default": "no"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("s3_uris",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "image/output"

    def process_metadata(self, custom_metadata, is_base64):
        """
        Process metadata string, handling both regular strings and base64 content.
        Returns tuple of (processed_content, is_binary)
        """
        if not custom_metadata:
            return None, False

        if is_base64 == "yes":
            try:
                # Try to decode base64 content
                decoded = base64.b64decode(custom_metadata)
                return decoded, True
            except Exception as e:
                logger.warning(
                    f"Failed to decode base64 metadata, treating as string: {e}"
                )
                return custom_metadata, False

        return custom_metadata, False

    def save_images(
        self,
        images,
        bucket,
        prefix,
        compression=4,
        include_workflow_metadata="enabled",
        custom_metadata="",
        metadata_is_base64="no",
        prompt=None,
        extra_pnginfo=None,
    ):
        """
        Save images to S3 with flexible metadata support
        """
        try:
            s3_client = get_s3_client()
            s3_uris = []
            results = []

            # Process metadata based on type
            metadata = PngInfo()

            # Add workflow metadata if enabled
            if include_workflow_metadata == "enabled":
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            # Add custom metadata if provided
            if custom_metadata:
                processed_metadata, is_binary = self.process_metadata(
                    custom_metadata, metadata_is_base64
                )
                if processed_metadata is not None:
                    if is_binary:
                        # For binary data (like decoded base64), store as binary
                        metadata.add_text("custom_metadata_binary", "true")
                        metadata.add_text(
                            "custom_metadata", processed_metadata.decode("latin-1")
                        )
                    else:
                        # For regular strings, store directly
                        metadata.add_text("custom_metadata_binary", "false")
                        metadata.add_text("custom_metadata", processed_metadata)

            for image in images:
                # Convert the image tensor to a PIL Image
                i = 255.0 * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

                # Generate a unique filename
                filename = f"{uuid.uuid4()}.png"
                s3_key = os.path.join(prefix, filename) if prefix else filename

                # Save to temporary file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".png"
                ) as temp_file:
                    try:
                        # Save with metadata and compression
                        img.save(
                            temp_file.name, pnginfo=metadata, compress_level=compression
                        )

                        # Upload to S3
                        s3_client.upload_file(temp_file.name, bucket, s3_key)

                        # Generate S3 URI
                        s3_uri = f"s3://{bucket}/{s3_key}"
                        s3_uris.append(s3_uri)

                        # Add to results
                        results.append(
                            {
                                "filename": filename,
                                "subfolder": prefix,
                                "type": "s3_output",
                                "uri": s3_uri,
                            }
                        )

                    finally:
                        # Cleanup
                        if os.path.exists(temp_file.name):
                            os.unlink(temp_file.name)

            return {"ui": {"images": results}, "result": (s3_uris,)}

        except Exception as e:
            logger.error(f"Failed to save images to S3: {e}")
            raise
