from .bezier_node import BezierMapping
from .hex_to_rgb import HexToRGB
from .was_fork import Molde_Text_Concatenate, Molde_Load_Image_Batch

# Initialize or extend the NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS dictionaries
NODE_CLASS_MAPPINGS = NODE_CLASS_MAPPINGS if "NODE_CLASS_MAPPINGS" in globals() else {}
NODE_DISPLAY_NAME_MAPPINGS = (
    NODE_DISPLAY_NAME_MAPPINGS if "NODE_DISPLAY_NAME_MAPPINGS" in globals() else {}
)

# Update the mappings for each node individually
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

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
