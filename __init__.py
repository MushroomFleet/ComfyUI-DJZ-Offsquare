"""
ComfyUI-DJZ-Offsquare - ComfyUI Custom Node Package
Creates optimized image collages from batches of 2-6 images.

Installation:
    Clone or copy this folder to: ComfyUI/custom_nodes/ComfyUI-DJZ-Offsquare/

Author: Drift Johnson
Repository: https://github.com/MushroomFleet/ComfyUI-DJZ-Offsquare
"""

from .DJZ_Offsquare import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Version info
__version__ = "1.0.0"
__author__ = "Drift Johnson"

# Optional: Web directory for custom JavaScript extensions
# WEB_DIRECTORY = "./web"
