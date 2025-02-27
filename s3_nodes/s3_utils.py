import os
import boto3
from dotenv import load_dotenv
from .logger import logger

# Load environment variables immediately when module is imported
load_dotenv()


def get_s3_client():
    """
    Create an S3 client using environment variables.
    """
    try:
        # Get required configuration from environment variables
        region = os.getenv("S3_REGION")
        access_key = os.getenv("S3_ACCESS_KEY")
        secret_key = os.getenv("S3_SECRET_KEY")

        # Validate that all required variables are present
        if not all([region, access_key, secret_key]):
            err = "Missing required S3 environment variables (S3_REGION, S3_ACCESS_KEY, S3_SECRET_KEY)"
            logger.error(err)
            raise ValueError(err)

        # Create and return the S3 client
        return boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    except Exception as e:
        err = f"Failed to create S3 client: {e}"
        logger.error(err)
        raise


def parse_s3_uri(uri):
    """
    Parse an S3 URI into bucket name and key.
    """
    if not uri.startswith("s3://"):
        raise ValueError("URI must start with 's3://'")

    path = uri[5:]
    parts = path.split("/", 1)

    if len(parts) != 2:
        raise ValueError("Invalid S3 URI format. Expected s3://bucket/key")

    bucket = parts[0]
    key = parts[1]

    if not bucket:
        raise ValueError("No bucket specified in URI")

    return bucket, key


def s3_path_join(*args):
    """
    Join path components for S3 keys with forward slashes, ignoring empty components.

    Args:
        *args: Path components to join.

    Returns:
        str: Joined path using forward slashes (S3 standard).
    """
    # Filter out empty strings and join with forward slash
    return "/".join(arg.strip("/") for arg in args if arg)


def is_zip_file(s3_client, bucket, key):
    """
    Check if an S3 object is a zip file by examining its content type or by checking the file extension.

    Args:
        s3_client: Boto3 S3 client instance
        bucket (str): S3 bucket name
        key (str): S3 object key

    Returns:
        bool: True if the object is likely a zip file, False otherwise
    """
    try:
        # Check by file extension first (faster)
        if key.lower().endswith(".zip"):
            return True

        # If extension check is inconclusive, check content type
        head_response = s3_client.head_object(Bucket=bucket, Key=key)
        content_type = head_response.get("ContentType", "")

        return content_type.lower() in [
            "application/zip",
            "application/x-zip-compressed",
        ]
    except Exception as e:
        logger.error(f"Error checking if object is a zip file: {e}")
        return False
