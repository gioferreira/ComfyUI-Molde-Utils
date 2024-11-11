class HexToRGB:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hex_color": ("STRING", {"default": "#FF0000"}),  # Default to red color
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("red", "green", "blue")
    FUNCTION = "hex_to_rgb"
    CATEGORY = "Molde Utilities"

    @staticmethod
    def hex_to_rgb(hex_color):
        if hex_color.startswith("#"):
            hex_color = hex_color[1:]

        red = int(hex_color[0:2], 16)
        green = int(hex_color[2:4], 16)
        blue = int(hex_color[4:6], 16)

        return red, green, blue
