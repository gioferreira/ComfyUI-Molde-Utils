import os
import uuid
import boto3
from urllib.parse import urlparse
from .logger import logger


def parse_s3_uri(uri):
    """
    Parse an S3 URI into bucket name and key.

    Args:
        uri (str): S3 URI in the format s3://bucket/path/to/file

    Returns:
        tuple: (bucket_name, key)

    Raises:
        ValueError: If URI format is invalid
    """
    if not uri.startswith("s3://"):
        raise ValueError("URI must start with 's3://'")

    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    if not bucket:
        raise ValueError("No bucket specified in URI")

    return bucket, key


def get_s3_client():
    """
    Create an S3 client using environment variables.

    Required environment variables:
    - S3_REGION
    - S3_ACCESS_KEY
    - S3_SECRET_KEY

    Returns:
        boto3.client: Configured S3 client
    """
    try:
        region = os.getenv("S3_REGION")
        access_key = os.getenv("S3_ACCESS_KEY")
        secret_key = os.getenv("S3_SECRET_KEY")

        if not all([region, access_key, secret_key]):
            raise ValueError("Missing required S3 environment variables")

        return boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}")
        raise


def list_s3_images(bucket, prefix=""):
    """
    List all image files in an S3 bucket under the given prefix.

    Args:
        bucket (str): S3 bucket name
        prefix (str): Path prefix to filter by

    Returns:
        list: List of image file paths
    """
    try:
        s3_client = get_s3_client()
        paginator = s3_client.get_paginator("list_objects_v2")
        image_extensions = (".png", ".jpg", ".jpeg", ".webp")

        # Make sure prefix doesn't start with '/' and ends with '/'
        prefix = prefix.strip("/")
        if prefix:
            prefix = prefix + "/"

        image_files = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                if key.lower().endswith(image_extensions):
                    image_files.append(key)

        return sorted(image_files) if image_files else [""]
    except Exception as e:
        logger.error(f"Failed to list S3 images: {e}")
        return [""]


def generate_s3_uri(bucket, key):
    """
    Generate a properly formatted S3 URI.

    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key

    Returns:
        str: Formatted S3 URI
    """
    return f"s3://{bucket}/{key}"


def generate_unique_filename(original_filename):
    """
    Generate a unique filename using UUID while preserving the original extension.

    Args:
        original_filename (str): Original filename

    Returns:
        str: New filename with UUID
    """
    ext = os.path.splitext(original_filename)[1]
    return f"{uuid.uuid4()}{ext}"
