# Import existing nodes
from .bezier_node import BezierMapping
from .hex_to_rgb import HexToRGB
from .was_fork import Molde_Text_Concatenate, Molde_Load_Image_Batch

# Import new S3 nodes
from .s3_nodes.api_load_image import LoadImageS3API
from .s3_nodes.api_save_image import SaveImageS3API

# Initialize or extend the NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS dictionaries
NODE_CLASS_MAPPINGS = NODE_CLASS_MAPPINGS if "NODE_CLASS_MAPPINGS" in globals() else {}
NODE_DISPLAY_NAME_MAPPINGS = (
    NODE_DISPLAY_NAME_MAPPINGS if "NODE_DISPLAY_NAME_MAPPINGS" in globals() else {}
)

# Update the mappings for existing nodes
NODE_CLASS_MAPPINGS.update(
    {
        "BezierMapping": BezierMapping,
    }
)
NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "Bezier Mapping": "BezierMapping",
    }
)

NODE_CLASS_MAPPINGS.update(
    {
        "HexToRGB": HexToRGB,
    }
)
NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "HEX to RGB": "HexToRGB",
    }
)

NODE_CLASS_MAPPINGS.update(
    {
        "TextConcatenate": Molde_Text_Concatenate,
    }
)
NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "Text Concatenate": "TextConcatenate",
    }
)

NODE_CLASS_MAPPINGS.update(
    {
        "LoadImageBatch": Molde_Load_Image_Batch,
    }
)
NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "Load Image Batch": "LoadImageBatch",
    }
)

# Add new S3 nodes to the mappings
NODE_CLASS_MAPPINGS.update(
    {
        "LoadImageS3API": LoadImageS3API,
        "SaveImageS3API": SaveImageS3API,
    }
)
NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "Load Image from S3 (URI)": "LoadImageS3API",
        "Save Image to S3": "SaveImageS3API",
    }
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
