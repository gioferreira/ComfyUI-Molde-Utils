import os
import json

# import uuid
import tempfile
import zipfile
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from .s3_utils import get_s3_client, s3_path_join
from .logger import logger
import folder_paths  # type: ignore


class SaveZipS3API:
    def __init__(self):
        self.compress_level = 4
        self.type = "output"  # This helps ComfyUI identify this as an output node
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "bucket": ("STRING", {"default": ""}),
                "prefix": ("STRING", {"default": ""}),
                "archive_name": ("STRING", {"default": "images.zip"}),
                "image_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "compression": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 9,
                        "step": 1,
                        "tooltip": "0 = no compression (fast/large), 9 = max compression (slow/small)",
                    },
                ),
                "include_workflow_metadata": (
                    ["enabled", "disabled"],
                    {"default": "disabled"},
                ),
            },
            "optional": {
                "custom_metadata": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Custom metadata string or base64 content",
                    },
                ),
                "metadata_is_base64": (["yes", "no"], {"default": "yes"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("s3_uri",)
    FUNCTION = "save_zip"
    OUTPUT_NODE = True
    CATEGORY = "image/output"

    def save_zip(
        self,
        images,
        bucket,
        prefix,
        archive_name,
        image_format="png",
        compression=4,
        include_workflow_metadata="enabled",
        custom_metadata="",
        metadata_is_base64="no",
        prompt=None,
        extra_pnginfo=None,
    ):
        """Save images to a ZIP file and upload to S3"""
        try:
            s3_client = get_s3_client()
            temp_dir = None
            temp_zip = None

            try:
                # Create a temporary directory to store images
                temp_dir = tempfile.mkdtemp()

                # Create PngInfo for metadata
                metadata = PngInfo()

                # Add custom metadata first (raw, without any processing)
                if custom_metadata:
                    metadata.add_text("metadata", custom_metadata)
                    if metadata_is_base64 == "yes":
                        metadata.add_text("metadata_is_base64", "yes")

                # Add workflow metadata if enabled (after custom metadata)
                if include_workflow_metadata == "enabled":
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                # Process each image
                for idx, image in enumerate(images):
                    # Convert the image tensor to a PIL Image
                    i = 255.0 * image.cpu().numpy()
                    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

                    # Generate a filename
                    if image_format == "png":
                        filename = f"image_{idx:05d}.png"
                        img.save(
                            os.path.join(temp_dir, filename),
                            format="PNG",
                            pnginfo=metadata,
                            compress_level=compression,
                        )
                    elif image_format == "jpeg":
                        filename = f"image_{idx:05d}.jpg"
                        img.save(
                            os.path.join(temp_dir, filename),
                            format="JPEG",
                            quality=int(100 - (compression * 10)),
                        )
                    elif image_format == "webp":
                        filename = f"image_{idx:05d}.webp"
                        img.save(
                            os.path.join(temp_dir, filename),
                            format="WEBP",
                            quality=int(100 - (compression * 10)),
                        )

                    img.close()

                # Create a temporary zip file
                temp_zip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
                temp_zip.close()

                # Create the ZIP file
                with zipfile.ZipFile(
                    temp_zip.name, "w", zipfile.ZIP_DEFLATED, compresslevel=compression
                ) as zipf:
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, temp_dir)
                            zipf.write(file_path, arcname)

                # Ensure archive_name ends with .zip
                if not archive_name.lower().endswith(".zip"):
                    archive_name += ".zip"

                # Create the S3 key
                s3_key = s3_path_join(prefix, archive_name) if prefix else archive_name

                # Upload to S3
                s3_client.upload_file(temp_zip.name, bucket, s3_key)

                # Generate S3 URI
                s3_uri = f"s3://{bucket}/{s3_key}"

                # Print URI to console
                logger.info(f"Saved ZIP file to: {s3_uri}")

                result = {
                    "ui": {
                        "zip": [
                            {
                                "filename": archive_name,
                                "type": "s3_output",
                                "uri": s3_uri,
                            }
                        ]
                    },
                    "result": (s3_uri,),
                }

                return result

            finally:
                # Clean up resources
                if temp_dir and os.path.exists(temp_dir):
                    try:
                        for root, dirs, files in os.walk(temp_dir, topdown=False):
                            for file in files:
                                os.unlink(os.path.join(root, file))
                            for dir_name in dirs:
                                os.rmdir(os.path.join(root, dir_name))
                        os.rmdir(temp_dir)
                    except (OSError, PermissionError) as e:
                        logger.warning(
                            f"Failed to delete temporary directory {temp_dir}: {e}"
                        )

                if temp_zip and os.path.exists(temp_zip.name):
                    try:
                        os.unlink(temp_zip.name)
                    except (OSError, PermissionError) as e:
                        logger.warning(
                            f"Failed to delete temporary zip file {temp_zip.name}: {e}"
                        )

        except Exception as e:
            logger.error(f"Failed to save ZIP to S3: {e}")
            raise
