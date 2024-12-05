import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import tempfile
import os
from .s3_utils import get_s3_client, parse_s3_uri
from .logger import logger

class LoadImageS3API:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "s3_uri": ("STRING", {
                    "default": "s3://bucket-name/path/to/image.png",
                    "multiline": False,
                    "placeholder": "Enter S3 URI (s3://bucket/path/to/image.png)"
                })
            }
        }
    
    CATEGORY = "image/input"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    
    def load_image(self, s3_uri):
        """
        Load an image from an S3 URI and return it as a tensor.
        
        Args:
            s3_uri (str): Full S3 URI (s3://bucket/path/to/image.png)
            
        Returns:
            tuple: (image_tensor, mask_tensor)
        """
        try:
            # Parse the S3 URI to get bucket and key
            bucket, key = parse_s3_uri(s3_uri)
            s3_client = get_s3_client()
            
            # Create a temporary file to store the downloaded image
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                try:
                    # Download the file from S3
                    s3_client.download_file(bucket, key, temp_file.name)
                    
                    # Process the image
                    img = Image.open(temp_file.name)
                    output_images = []
                    output_masks = []
                    
                    for i in ImageSequence.Iterator(img):
                        # Handle image orientation based on EXIF data
                        i = ImageOps.exif_transpose(i)
                        if i.mode == 'I':
                            i = i.point(lambda i: i * (1 / 255))
                        
                        # Convert image to RGB and normalize
                        image = i.convert("RGB")
                        image = np.array(image).astype(np.float32) / 255.0
                        image = torch.from_numpy(image)[None,]
                        
                        # Handle alpha channel if present
                        if 'A' in i.getbands():
                            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                            mask = 1. - torch.from_numpy(mask)
                        else:
                            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
                        
                        output_images.append(image)
                        output_masks.append(mask.unsqueeze(0))
                    
                    # Handle single images and image sequences
                    if len(output_images) > 1:
                        output_image = torch.cat(output_images, dim=0)
                        output_mask = torch.cat(output_masks, dim=0)
                    else:
                        output_image = output_images[0]
                        output_mask = output_masks[0]
                    
                    return (output_image, output_mask)
                    
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                
        except Exception as e:
            logger.error(f"Failed to load image from S3 URI '{s3_uri}': {e}")
            raise
    
    @classmethod
    def VALIDATE_INPUTS(s, s3_uri):
        """
        Validate the S3 URI format.
        """
        try:
            if not s3_uri:
                return "S3 URI is required"
                
            # Test parsing the URI
            parse_s3_uri(s3_uri)
            return True
            
        except Exception as e:
            return str(e)