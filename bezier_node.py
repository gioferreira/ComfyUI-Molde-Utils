# import torch
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms

# Use 'Agg' backend to avoid GUI issues on headless systems
plt.switch_backend("Agg")


class BezierMapping:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "t_value": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),  # Input t
                "P0": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "P1": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "P2": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "P3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "output_min": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "output_max": ("FLOAT", {"default": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("FLOAT", "IMAGE")
    RETURN_NAMES = ("Mapped y-Value", "Bezier Curve Image")
    FUNCTION = "compute"
    CATEGORY = "Molde Utilities"

    def compute(self, t_value, P0, P1, P2, P3, output_min, output_max):
        def bezier(t, P0, P1, P2, P3):
            return (
                (1 - t) ** 3 * P0
                + 3 * (1 - t) ** 2 * t * P1
                + 3 * (1 - t) * t**2 * P2
                + t**3 * P3
            )

        # Calculate the y_value using the given t_value
        y_value = bezier(t_value, P0, P1, P2, P3)

        # Map the y_value to the output range [output_min, output_max]
        mapped_y = output_min + (output_max - output_min) * y_value

        # Generate the plot of the curve
        t_values = np.linspace(0, 1, 100)
        y_points = [
            output_min + (output_max - output_min) * bezier(t, P0, P1, P2, P3)
            for t in t_values
        ]

        # Create a new figure and plot
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(t_values, y_points, color="black", linewidth=1.5)
        ax.scatter(
            [t_value], [mapped_y], color="red", zorder=5
        )  # Highlight the input/output point
        ax.set_title("Bezier Curve (t vs y)")
        ax.set_xlabel("t")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.set_xlim(0, 1)
        ax.set_ylim(min(output_min, output_max), max(output_min, output_max))
        plt.tight_layout()

        # Save plot to a BytesIO object instead of a file
        buffer_io = BytesIO()
        plt.savefig(buffer_io, format="png", bbox_inches="tight")
        plt.close(fig)  # Close the figure to ensure no additional plots are generated

        # Load image from BytesIO and convert to tensor
        buffer_io.seek(0)
        img = Image.open(buffer_io)
        img_tensor = transforms.ToTensor()(img)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.permute(
            [0, 2, 3, 1]
        )  # Adjust to match your expected format

        return mapped_y, img_tensor
