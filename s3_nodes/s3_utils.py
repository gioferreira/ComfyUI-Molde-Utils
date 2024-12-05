import os
import boto3
from .logger import logger
from dotenv import load_dotenv


def load_environment_variables():
    """
    Load environment variables from .env files in multiple locations.
    Checks both the custom node directory and the ComfyUI root directory.
    """
    # Get the current file's directory (custom node directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the ComfyUI root directory (two levels up from s3_nodes)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

    # Try loading from custom node directory first
    if os.path.exists(os.path.join(current_dir, ".env")):
        load_dotenv(os.path.join(current_dir, ".env"))
        logger.info("Loaded .env from custom node directory")

    # Then try loading from ComfyUI root directory
    if os.path.exists(os.path.join(root_dir, ".env")):
        load_dotenv(os.path.join(root_dir, ".env"))
        logger.info("Loaded .env from ComfyUI root directory")


def get_s3_client():
    """
    Create an S3 client using environment variables.
    Now with better environment variable loading and error handling.
    """
    # Load environment variables from all possible locations
    load_environment_variables()

    try:
        # Get required configuration from environment variables
        region = os.getenv("S3_REGION")
        access_key = os.getenv("S3_ACCESS_KEY")
        secret_key = os.getenv("S3_SECRET_KEY")

        # Validate that all required variables are present
        missing_vars = []
        if not region:
            missing_vars.append("S3_REGION")
        if not access_key:
            missing_vars.append("S3_ACCESS_KEY")
        if not secret_key:
            missing_vars.append("S3_SECRET_KEY")

        if missing_vars:
            err = (
                f"Missing required S3 environment variables: {', '.join(missing_vars)}"
            )
            logger.error(err)
            logger.error(
                f"Please ensure these variables are set in your .env file at either:"
            )
            logger.error(f"- ComfyUI root directory")
            logger.error(f"- Custom node directory")
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

    # Remove the s3:// prefix and split into bucket and key
    path = uri[5:]
    parts = path.split("/", 1)

    if len(parts) != 2:
        raise ValueError("Invalid S3 URI format. Expected s3://bucket/key")

    bucket = parts[0]
    key = parts[1]

    if not bucket:
        raise ValueError("No bucket specified in URI")

    return bucket, key
