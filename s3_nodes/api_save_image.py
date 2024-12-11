import os
import json
import uuid
import tempfile
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from .s3_utils import get_s3_client
from .logger import logger
import folder_paths  # type: ignore


class SaveImageS3API:
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
        metadata_is_base64="no",
        prompt=None,
        extra_pnginfo=None,
    ):
        """Save images to S3 with metadata support"""
        try:
            s3_client = get_s3_client()
            s3_uris = []
            results = []

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

            for image in images:
                # Convert the image tensor to a PIL Image
                i = 255.0 * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

                # Generate a unique filename
                filename = f"{uuid.uuid4()}.png"
                s3_key = os.path.join(prefix, filename) if prefix else filename

                temp_file = None
                temp_filename = None

                try:
                    # Create and immediately close the temporary file to get its name
                    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    temp_filename = temp_file.name
                    temp_file.close()

                    # Save with metadata and compression
                    img.save(
                        temp_filename,
                        format="PNG",
                        pnginfo=metadata,
                        compress_level=compression,
                    )

                    # Upload to S3
                    s3_client.upload_file(temp_filename, bucket, s3_key)

                    # Generate S3 URI
                    s3_uri = f"s3://{bucket}/{s3_key}"
                    s3_uris.append(s3_uri)

                    # Print URI to console
                    print(f"Saved image to: {s3_uri}")

                    # Add to results for UI
                    results.append(
                        {
                            "filename": filename,
                            "subfolder": prefix,
                            "type": "s3_output",
                            "uri": s3_uri,
                            "direct": True,
                        }
                    )

                finally:
                    # Clean up resources
                    if img:
                        img.close()

                    # Try to delete the temporary file
                    if temp_filename and os.path.exists(temp_filename):
                        try:
                            os.unlink(temp_filename)
                        except (OSError, PermissionError) as e:
                            logger.warning(
                                f"Failed to delete temporary file {temp_filename}: {e}"
                            )

            # # After all images are saved, write URIs to output file
            # txt_filename = f"{uuid.uuid4()}_s3_uris.txt"
            # txt_path = os.path.join(self.output_dir, txt_filename)

            # with open(txt_path, "w") as f:
            #     json.dump(s3_uris, f)

            # # Add txt file to results
            # results.append(
            #     {
            #         "filename": txt_filename,
            #         "subfolder": "",
            #         "type": "output",
            #         "uri_file": True,
            #     }
            # )

            resposta = {
                "ui": {"images": results},
                "workflow": {"output_type": "s3", "uris": s3_uris},
                "result": (s3_uris,),
            }
            print(f"{resposta=}")

            return resposta

        except Exception as e:
            logger.error(f"Failed to save images to S3: {e}")
            raise
