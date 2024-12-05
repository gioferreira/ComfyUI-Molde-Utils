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
