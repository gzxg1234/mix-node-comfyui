import os
import re
import folder_paths as comfy_paths
import comfy.diffusers_convert
import comfy.samplers
import comfy.sd
import comfy.utils
import comfy.clip_vision


class MIX_Lora_Parser:
    def __init__(self):
        self.loaded_lora = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("MODEL", "CLIP")
    FUNCTION = "parse_lora"

    CATEGORY = "MIX_NODE/Loaders"

    LORA_PATTERN = r"<lora:(\w+)(?::([+-]?\d+(\.\d+)?))?>"

    def parse_lora(self, model, clip, text):
        if not text:
            return (model, clip)
        lora_info_list = self.parse_lora_infos(text)
        for name, strength in lora_info_list.items():
            model, clip = self.load_lora(model, clip, name, strength)
        return (model, clip)

    def parse_lora_infos(self, text):
        lora_info_list = {}
        groups = text.split(",")
        for group in groups:
            match = re.search(self.LORA_PATTERN, group)
            if match:
                lora_name = match.group(1)
                lora_strength = match.group(2)
                if lora_strength:
                    lora_strength = float(lora_strength)
                else:
                    lora_strength = 1
                lora_info_list[lora_name] = lora_strength
        return lora_info_list

    def load_lora(self, model, clip, lora_name, stength):
        if not lora_name:
            return (model, clip)

        # search file
        for x in comfy_paths.folder_names_and_paths["loras"][0]:
            lora_path = self.search_lora_file(x, lora_name)
            if lora_path is not None:
                break

        if lora_path is None:
            print(f"lora {lora_name} not find")
            return (model, clip)

        lora_cache = self.loaded_lora.get(lora_name)
        lora = None
        if lora_cache:
            if lora_cache[0] == lora_path:
                lora = lora_cache[1]
            else:
                del lora_cache

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora[lora_name] = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora, stength, stength
        )
        return (model_lora, clip_lora)

    def search_lora_file(self, path, name):
        if os.path.isdir(path):
            for sub in os.listdir(path):
                result = self.search_lora_file(os.path.join(path, sub), name)
                if result:
                    return result
        elif os.path.isfile(path):
            baseName = os.path.basename(path)
            file_name, file_ext = os.path.splitext(baseName)
            if file_name == name and file_ext in [
                ".ckpt",
                ".pt",
                ".pth",
                ".safetensors",
            ]:
                return path


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "MIX_Lora_Parser": MIX_Lora_Parser,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MIX_Lora_Parser": "MIX_Lora_Parser",
}
