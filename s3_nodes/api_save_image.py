import os
import json
import uuid
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
                "custom_metadata": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Custom metadata as JSON string",
                    },
                ),
            },
            # These hidden inputs receive workflow metadata from ComfyUI
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("s3_uris",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "image/output"

    def save_images(
        self,
        images,
        bucket,
        prefix,
        compression=4,
        include_workflow_metadata="enabled",
        custom_metadata="",
        prompt=None,
        extra_pnginfo=None,
    ):
        """
        Save images to S3 with metadata support

        Args:
            images: Image tensors to save
            bucket: S3 bucket name
            prefix: Path prefix in bucket
            compression: PNG compression level (0-9)
            include_workflow_metadata: Whether to include ComfyUI's workflow metadata
            custom_metadata: Optional custom metadata as JSON string
            prompt: Workflow prompt info (from ComfyUI)
            extra_pnginfo: Extra workflow info (from ComfyUI)
        """
        try:
            s3_client = get_s3_client()
            s3_uris = []
            results = []

            # Prepare metadata
            metadata = None
            if include_workflow_metadata == "enabled" or custom_metadata:
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
                    try:
                        # Validate custom metadata is proper JSON
                        custom_data = json.loads(custom_metadata)
                        metadata.add_text("custom_metadata", json.dumps(custom_data))
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Invalid JSON in custom_metadata, skipping: {e}"
                        )

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
                        # Save with metadata and compression
                        img.save(
                            temp_file.name, pnginfo=metadata, compress_level=compression
                        )

                        # Upload to S3
                        s3_client.upload_file(temp_file.name, bucket, s3_key)

                        # Generate and store the S3 URI
                        s3_uri = f"s3://{bucket}/{s3_key}"
                        s3_uris.append(s3_uri)

                        # Add to results for UI
                        results.append(
                            {
                                "filename": filename,
                                "subfolder": prefix,
                                "type": "s3_output",
                                "uri": s3_uri,
                            }
                        )

                    finally:
                        # Clean up the temporary file
                        if os.path.exists(temp_file.name):
                            os.unlink(temp_file.name)

            # Return both UI info and the S3 URIs
            # The UI info will be used by ComfyUI
            # The s3_uris will be available in the API response
            return {"ui": {"images": results}, "result": (s3_uris,)}

        except Exception as e:
            logger.error(f"Failed to save images to S3: {e}")
            raise
