"""
ComfyUI-DJZ-Offsquare - ComfyUI Custom Node Package
Creates optimized image collages from batches of 2-6 images.

Installation:
    Clone or copy this folder to: ComfyUI/custom_nodes/ComfyUI-DJZ-Offsquare/

Author: Drift Johnson
Repository: https://github.com/MushroomFleet/ComfyUI-DJZ-Offsquare
"""

from .DJZ_Offsquare import NODE_CLASS_MAPPINGS as DJZ_Offsquare_MAPPINGS
from .DJZ_Offsquare import NODE_DISPLAY_NAME_MAPPINGS as DJZ_Offsquare_DISPLAY
from .DJZ_Offsquare_V2 import NODE_CLASS_MAPPINGS as DJZ_Offsquare_V2_MAPPINGS
from .DJZ_Offsquare_V2 import NODE_DISPLAY_NAME_MAPPINGS as DJZ_Offsquare_V2_DISPLAY

NODE_CLASS_MAPPINGS = {**DJZ_Offsquare_MAPPINGS, **DJZ_Offsquare_V2_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**DJZ_Offsquare_DISPLAY, **DJZ_Offsquare_V2_DISPLAY}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Version info
__version__ = "1.0.0"
__author__ = "Drift Johnson"

# Optional: Web directory for custom JavaScript extensions
# WEB_DIRECTORY = "./web"
