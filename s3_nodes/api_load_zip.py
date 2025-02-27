import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import tempfile
import os
import zipfile
import io
from .s3_utils import get_s3_client, parse_s3_uri
from .logger import logger


class LoadZipS3API:
    s3_client = None

    # Supported image extensions for extraction
    SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"]

    @classmethod
    def get_s3_client(cls):
        """
        Get or create the S3 client when needed, following the singleton pattern.
        This lazy initialization helps avoid issues during module loading.
        """
        if cls.s3_client is None:
            cls.s3_client = get_s3_client()
        return cls.s3_client

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "s3_uri": (
                    "STRING",
                    {
                        "default": "s3://bucket-name/path/to/archive.zip",
                        "multiline": False,
                        "placeholder": "Enter S3 URI (s3://bucket/path/to/archive.zip)",
                    },
                ),
                "frame_load_cap": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10000, "step": 1},
                ),
                "skip_first_images": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10000, "step": 1},
                ),
                "select_every_nth": (
                    "INT",
                    {"default": 1, "min": 1, "max": 100, "step": 1},
                ),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    CATEGORY = "image/input"
    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "frame_count")
    FUNCTION = "load_zip"

    def load_zip(
        self,
        s3_uri,
        frame_load_cap=0,
        skip_first_images=0,
        select_every_nth=1,
        meta_batch=None,
        unique_id=None,
    ):
        """
        Load a ZIP file from an S3 URI, extract images, and return them as tensors.
        """
        temp_dir = None
        try:
            # Parse the S3 URI to get bucket and key
            bucket, key = parse_s3_uri(s3_uri)
            s3_client = self.get_s3_client()

            # Create a temporary directory to store the extracted files
            temp_dir = tempfile.mkdtemp()

            # Create a temporary file to store the downloaded zip
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_filename = temp_file.name
            temp_file.close()

            logger.info(f"Downloading ZIP file from S3: {s3_uri}")
            # Download the file from S3
            s3_client.download_file(bucket, key, temp_filename)

            # Verify it's a valid ZIP file
            if not zipfile.is_zipfile(temp_filename):
                raise ValueError(f"The file at {s3_uri} is not a valid ZIP file")

            # Extract the ZIP file contents
            logger.info(f"Extracting ZIP file to temporary directory: {temp_dir}")
            with zipfile.ZipFile(temp_filename, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            # Find all image files in the extracted directory
            image_files = []
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if any(
                        file.lower().endswith(ext) for ext in self.SUPPORTED_EXTENSIONS
                    ):
                        image_files.append(os.path.join(root, file))

            # Sort the files alphabetically
            image_files.sort()

            # Apply frame selection parameters
            if skip_first_images > 0:
                image_files = image_files[skip_first_images:]

            if select_every_nth > 1:
                image_files = image_files[::select_every_nth]

            if frame_load_cap > 0:
                image_files = image_files[:frame_load_cap]

            if not image_files:
                raise ValueError(
                    f"No valid image files found in the ZIP archive at {s3_uri}"
                )

            logger.info(f"Found {len(image_files)} images in ZIP file")

            # Process images
            output_images = []
            output_masks = []

            # Determine the most common image size for consistency
            sizes = {}
            for image_path in image_files:
                with Image.open(image_path) as img:
                    img = ImageOps.exif_transpose(img)
                    size = img.size
                    sizes[size] = sizes.get(size, 0) + 1

            # Get most common size
            common_size = max(sizes.items(), key=lambda x: x[1])[0]
            width, height = common_size

            # Load and process each image
            for image_path in image_files:
                with Image.open(image_path) as img:
                    img = ImageOps.exif_transpose(img)

                    # Resize if necessary to match the most common size
                    if img.size != common_size:
                        img = img.resize(common_size, Image.Resampling.LANCZOS)

                    # Convert to RGB
                    if img.mode == "I":
                        img = img.point(lambda i: i * (1 / 255))

                    image = img.convert("RGB")
                    image_array = np.array(image).astype(np.float32) / 255.0
                    output_images.append(torch.from_numpy(image_array))

                    # Extract alpha channel if it exists, otherwise create empty mask
                    if "A" in img.getbands():
                        mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
                        mask = 1.0 - torch.from_numpy(mask)
                    else:
                        mask = torch.zeros(
                            (height, width), dtype=torch.float32, device="cpu"
                        )

                    output_masks.append(mask)

            # Stack images and masks
            if output_images:
                output_image = torch.stack(output_images)
                output_mask = torch.stack(output_masks)
                frame_count = len(output_images)
                return (output_image, output_mask, frame_count)
            else:
                raise RuntimeError("Failed to process any images from the ZIP file")

        except Exception as e:
            logger.error(f"Failed to load ZIP from S3 URI '{s3_uri}': {e}")
            raise

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

            # Remove the temporary zip file
            if "temp_filename" in locals() and os.path.exists(temp_filename):
                try:
                    os.unlink(temp_filename)
                except (OSError, PermissionError) as e:
                    logger.warning(
                        f"Failed to delete temporary file {temp_filename}: {e}"
                    )
