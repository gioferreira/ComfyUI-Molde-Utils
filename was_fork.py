from PIL import (
    Image,
    ImageOps,
)
import glob
import os
import random
import torch
import numpy as np
import hashlib


class cstr(str):
    class color:
        END = "\33[0m"
        BOLD = "\33[1m"
        ITALIC = "\33[3m"
        UNDERLINE = "\33[4m"
        BLINK = "\33[5m"
        BLINK2 = "\33[6m"
        SELECTED = "\33[7m"

        BLACK = "\33[30m"
        RED = "\33[31m"
        GREEN = "\33[32m"
        YELLOW = "\33[33m"
        BLUE = "\33[34m"
        VIOLET = "\33[35m"
        BEIGE = "\33[36m"
        WHITE = "\33[37m"

        BLACKBG = "\33[40m"
        REDBG = "\33[41m"
        GREENBG = "\33[42m"
        YELLOWBG = "\33[43m"
        BLUEBG = "\33[44m"
        VIOLETBG = "\33[45m"
        BEIGEBG = "\33[46m"
        WHITEBG = "\33[47m"

        GREY = "\33[90m"
        LIGHTRED = "\33[91m"
        LIGHTGREEN = "\33[92m"
        LIGHTYELLOW = "\33[93m"
        LIGHTBLUE = "\33[94m"
        LIGHTVIOLET = "\33[95m"
        LIGHTBEIGE = "\33[96m"
        LIGHTWHITE = "\33[97m"

        GREYBG = "\33[100m"
        LIGHTREDBG = "\33[101m"
        LIGHTGREENBG = "\33[102m"
        LIGHTYELLOWBG = "\33[103m"
        LIGHTBLUEBG = "\33[104m"
        LIGHTVIOLETBG = "\33[105m"
        LIGHTBEIGEBG = "\33[106m"
        LIGHTWHITEBG = "\33[107m"

        @staticmethod
        def add_code(name, code):
            if not hasattr(cstr.color, name.upper()):
                setattr(cstr.color, name.upper(), code)
            else:
                raise ValueError(
                    f"'cstr' object already contains a code with the name '{name}'."
                )

    def __new__(cls, text):
        return super().__new__(cls, text)

    def __getattr__(self, attr):
        if attr.lower().startswith("_cstr"):
            code = getattr(self.color, attr.upper().lstrip("_cstr"))
            modified_text = self.replace(f"__{attr[1:]}__", f"{code}")
            return cstr(modified_text)
        elif attr.upper() in dir(self.color):
            code = getattr(self.color, attr.upper())
            modified_text = f"{code}{self}{self.color.END}"
            return cstr(modified_text)
        elif attr.lower() in dir(cstr):
            return getattr(cstr, attr.lower())
        else:
            raise AttributeError(f"'cstr' object has no attribute '{attr}'")

    def print(self, **kwargs):
        print(self, **kwargs)


# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


ALLOWED_EXT = (".jpeg", ".jpg", ".png", ".tiff", ".gif", ".bmp", ".webp")


# SHA-256 Hash
def get_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


# Text Concatenate
class Molde_Text_Concatenate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "delimiter": ("STRING", {"default": ", "}),
                "clean_whitespace": (["true", "false"],),
            },
            "optional": {
                "text_a": ("STRING", {"forceInput": True}),
                "text_b": ("STRING", {"forceInput": True}),
                "text_c": ("STRING", {"forceInput": True}),
                "text_d": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "tet_concatenate"

    CATEGORY = "Molde Utilities"

    def text_concatenate(self, delimiter, clean_whitespace, **kwargs):
        text_inputs = []

        # Handle special case where delimiter is "\n" (literal newline).
        if delimiter in ("\n", "\\n"):
            delimiter = "\n"

        # Iterate over the received inputs in sorted order.
        for k in sorted(kwargs.keys()):
            v = kwargs[k]

            # Only process string input ports.
            if isinstance(v, str):
                if clean_whitespace == "true":
                    # Remove leading and trailing whitespace around this input.
                    v = v.strip()

                # Only use this input if it's a non-empty string, since it
                # never makes sense to concatenate totally empty inputs.
                # NOTE: If whitespace cleanup is disabled, inputs containing
                # 100% whitespace will be treated as if it's a non-empty input.
                if v != "":
                    text_inputs.append(v)

        # Merge the inputs. Will always generate an output, even if empty.
        merged_text = delimiter.join(text_inputs)

        return (merged_text,)


# LOAD IMAGE BATCH
class Molde_Load_Image_Batch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["single_image", "incremental_image", "random"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "index": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
                "label": ("STRING", {"default": "Batch 001", "multiline": False}),
                "path": ("STRING", {"default": "", "multiline": False}),
                "pattern": ("STRING", {"default": "*", "multiline": False}),
                "allow_RGBA_output": (["false", "true"],),
            },
            "optional": {
                "filename_text_extension": (["true", "false"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "filename_text")
    FUNCTION = "load_batch_images"

    CATEGORY = "WAS Suite/IO"

    def load_batch_images(
        self,
        path,
        pattern="*",
        index=0,
        mode="single_image",
        seed=0,
        label="Batch 001",
        allow_RGBA_output="false",
        filename_text_extension="true",
    ):
        allow_RGBA_output = allow_RGBA_output == "true"

        if not os.path.exists(path):
            return (None,)
        fl = self.BatchImageLoader(path, label, pattern)
        # new_paths = fl.image_paths
        if mode == "single_image":
            image, filename = fl.get_image_by_id(index)
            if image is None:
                cstr(f"No valid image was found for the inded `{index}`").error.print()
                return (None, None)
        elif mode == "incremental_image":
            image, filename = fl.get_next_image()
            if image is None:
                cstr(
                    "No valid image was found for the next ID. Did you remove images from the source directory?"
                ).error.print()
                return (None, None)
        else:
            random.seed(seed)
            newindex = int(random.random() * len(fl.image_paths))
            image, filename = fl.get_image_by_id(newindex)
            if image is None:
                cstr(
                    "No valid image was found for the next ID. Did you remove images from the source directory?"
                ).error.print()
                return (None, None)

        # Update history
        # update_history_images(new_paths)

        if not allow_RGBA_output:
            image = image.convert("RGB")

        if filename_text_extension == "false":
            filename = os.path.splitext(filename)[0]

        return (pil2tensor(image), filename)

    class BatchImageLoader:
        def __init__(self, directory_path, label, pattern):
            # self.WDB = WDB
            self.image_paths = []
            self.load_images(directory_path, pattern)
            self.image_paths.sort()
            stored_directory_path = self.WDB.get("Batch Paths", label)
            stored_pattern = self.WDB.get("Batch Patterns", label)
            if stored_directory_path != directory_path or stored_pattern != pattern:
                self.index = 0
                self.WDB.insert("Batch Counters", label, 0)
                self.WDB.insert("Batch Paths", label, directory_path)
                self.WDB.insert("Batch Patterns", label, pattern)
            else:
                self.index = self.WDB.get("Batch Counters", label)
            self.label = label

        def load_images(self, directory_path, pattern):
            for file_name in glob.glob(
                os.path.join(glob.escape(directory_path), pattern), recursive=True
            ):
                if file_name.lower().endswith(ALLOWED_EXT):
                    abs_file_path = os.path.abspath(file_name)
                    self.image_paths.append(abs_file_path)

        def get_image_by_id(self, image_id):
            if image_id < 0 or image_id >= len(self.image_paths):
                cstr(f"Invalid image index `{image_id}`").error.print()
                return
            i = Image.open(self.image_paths[image_id])
            i = ImageOps.exif_transpose(i)
            return (i, os.path.basename(self.image_paths[image_id]))

        def get_next_image(self):
            if self.index >= len(self.image_paths):
                self.index = 0
            image_path = self.image_paths[self.index]
            self.index += 1
            if self.index == len(self.image_paths):
                self.index = 0
            cstr(
                f"{cstr.color.YELLOW}{self.label}{cstr.color.END} Index: {self.index}"
            ).msg.print()
            self.WDB.insert("Batch Counters", self.label, self.index)
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            return (i, os.path.basename(image_path))

        def get_current_image(self):
            if self.index >= len(self.image_paths):
                self.index = 0
            image_path = self.image_paths[self.index]
            return os.path.basename(image_path)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs["mode"] != "single_image":
            return float("NaN")
        else:
            fl = WAS_Load_Image_Batch.BatchImageLoader(
                kwargs["path"], kwargs["label"], kwargs["pattern"]
            )
            filename = fl.get_current_image()
            image = os.path.join(kwargs["path"], filename)
            sha = get_sha256(image)
            return sha
