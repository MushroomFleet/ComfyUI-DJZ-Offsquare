"""
DJZ_Offsquare_V2 - ComfyUI Custom Node
Creates optimized image collages from batches of 2-6 images.

Features:
- Multiple layout strategies per image count
- Aspect ratio selection (1:1, 16:9, 9:16, 3:2, 2:3, 4:3, 3:4)
- Customizable canvas total pixels (default 3072x3072)
- First image → top-left (bottom z-order)
- Last image → bottom-right with spill effect (top z-order)
- Aspect-ratio aware image placement

Author: Drift Johnson
Repository: https://github.com/MushroomFleet/ComfyUI-DJZ-Offsquare
"""

import torch
import numpy as np
from PIL import Image
import math

# =============================================================================
# LAYOUT STRATEGIES
# =============================================================================

LAYOUT_STRATEGIES = {
    2: [
        {
            "type": "diagonal",
            "regions": [
                {"x": 0, "y": 0, "w": 0.58, "h": 0.58, "z": 1},
                {"x": 0.38, "y": 0.38, "w": 0.62, "h": 0.62, "z": 2}
            ]
        },
        {
            "type": "vsplit",
            "regions": [
                {"x": 0, "y": 0, "w": 0.54, "h": 1, "z": 1},
                {"x": 0.48, "y": 0.02, "w": 0.52, "h": 0.98, "z": 2}
            ]
        },
        {
            "type": "hsplit",
            "regions": [
                {"x": 0, "y": 0, "w": 1, "h": 0.54, "z": 1},
                {"x": 0.02, "y": 0.48, "w": 0.98, "h": 0.52, "z": 2}
            ]
        }
    ],
    3: [
        {
            "type": "cascade",
            "regions": [
                {"x": 0, "y": 0, "w": 1, "h": 0.62, "z": 1},
                {"x": 0, "y": 0.44, "w": 0.52, "h": 0.56, "z": 2},
                {"x": 0.40, "y": 0.38, "w": 0.60, "h": 0.62, "z": 3}
            ]
        },
        {
            "type": "l-shape",
            "regions": [
                {"x": 0, "y": 0, "w": 0.70, "h": 0.70, "z": 1},
                {"x": 0, "y": 0.58, "w": 0.48, "h": 0.42, "z": 2},
                {"x": 0.46, "y": 0.38, "w": 0.54, "h": 0.62, "z": 3}
            ]
        },
        {
            "type": "reverse-l",
            "regions": [
                {"x": 0, "y": 0, "w": 0.58, "h": 1, "z": 1},
                {"x": 0.46, "y": 0, "w": 0.54, "h": 0.52, "z": 2},
                {"x": 0.44, "y": 0.42, "w": 0.56, "h": 0.58, "z": 3}
            ]
        }
    ],
    4: [
        {
            "type": "grid",
            "regions": [
                {"x": 0, "y": 0, "w": 0.54, "h": 0.54, "z": 1},
                {"x": 0.50, "y": 0, "w": 0.50, "h": 0.48, "z": 2},
                {"x": 0, "y": 0.50, "w": 0.48, "h": 0.50, "z": 3},
                {"x": 0.44, "y": 0.46, "w": 0.56, "h": 0.54, "z": 4}
            ]
        },
        {
            "type": "dominant-three",
            "regions": [
                {"x": 0, "y": 0, "w": 0.62, "h": 1, "z": 1},
                {"x": 0.58, "y": 0, "w": 0.42, "h": 0.34, "z": 2},
                {"x": 0.58, "y": 0.32, "w": 0.42, "h": 0.34, "z": 3},
                {"x": 0.54, "y": 0.62, "w": 0.46, "h": 0.38, "z": 4}
            ]
        },
        {
            "type": "t-layout",
            "regions": [
                {"x": 0, "y": 0, "w": 1, "h": 0.56, "z": 1},
                {"x": 0, "y": 0.52, "w": 0.34, "h": 0.48, "z": 2},
                {"x": 0.32, "y": 0.52, "w": 0.34, "h": 0.48, "z": 3},
                {"x": 0.60, "y": 0.48, "w": 0.40, "h": 0.52, "z": 4}
            ]
        }
    ],
    5: [
        {
            "type": "quincunx",
            "regions": [
                {"x": 0, "y": 0, "w": 0.54, "h": 0.54, "z": 1},
                {"x": 0.50, "y": 0, "w": 0.50, "h": 0.48, "z": 2},
                {"x": 0.36, "y": 0.36, "w": 0.28, "h": 0.28, "z": 3},
                {"x": 0, "y": 0.50, "w": 0.48, "h": 0.50, "z": 4},
                {"x": 0.44, "y": 0.46, "w": 0.56, "h": 0.54, "z": 5}
            ]
        },
        {
            "type": "asymmetric",
            "regions": [
                {"x": 0, "y": 0, "w": 0.56, "h": 0.56, "z": 1},
                {"x": 0.52, "y": 0, "w": 0.48, "h": 0.44, "z": 2},
                {"x": 0.38, "y": 0.38, "w": 0.24, "h": 0.24, "z": 3},
                {"x": 0, "y": 0.52, "w": 0.46, "h": 0.48, "z": 4},
                {"x": 0.40, "y": 0.44, "w": 0.60, "h": 0.56, "z": 5}
            ]
        },
        {
            "type": "l-shape",
            "regions": [
                {"x": 0, "y": 0, "w": 0.62, "h": 0.62, "z": 1},
                {"x": 0.58, "y": 0, "w": 0.42, "h": 0.48, "z": 2},
                {"x": 0, "y": 0.58, "w": 0.34, "h": 0.42, "z": 3},
                {"x": 0.32, "y": 0.58, "w": 0.32, "h": 0.42, "z": 4},
                {"x": 0.54, "y": 0.44, "w": 0.46, "h": 0.56, "z": 5}
            ]
        }
    ],
    6: [
        {
            "type": "grid-2x3",
            "regions": [
                {"x": 0, "y": 0, "w": 0.52, "h": 0.36, "z": 1},
                {"x": 0.48, "y": 0, "w": 0.52, "h": 0.34, "z": 2},
                {"x": 0, "y": 0.34, "w": 0.50, "h": 0.34, "z": 3},
                {"x": 0.48, "y": 0.32, "w": 0.52, "h": 0.34, "z": 4},
                {"x": 0, "y": 0.66, "w": 0.48, "h": 0.34, "z": 5},
                {"x": 0.44, "y": 0.62, "w": 0.56, "h": 0.38, "z": 6}
            ]
        },
        {
            "type": "grid-3x2",
            "regions": [
                {"x": 0, "y": 0, "w": 0.36, "h": 0.52, "z": 1},
                {"x": 0.34, "y": 0, "w": 0.34, "h": 0.50, "z": 2},
                {"x": 0.66, "y": 0, "w": 0.34, "h": 0.48, "z": 3},
                {"x": 0, "y": 0.48, "w": 0.34, "h": 0.52, "z": 4},
                {"x": 0.32, "y": 0.48, "w": 0.34, "h": 0.52, "z": 5},
                {"x": 0.60, "y": 0.44, "w": 0.40, "h": 0.56, "z": 6}
            ]
        },
        {
            "type": "feature-corner",
            "regions": [
                {"x": 0, "y": 0, "w": 0.56, "h": 0.56, "z": 1},
                {"x": 0.52, "y": 0, "w": 0.48, "h": 0.34, "z": 2},
                {"x": 0.52, "y": 0.32, "w": 0.48, "h": 0.34, "z": 3},
                {"x": 0, "y": 0.52, "w": 0.34, "h": 0.48, "z": 4},
                {"x": 0.32, "y": 0.52, "w": 0.34, "h": 0.48, "z": 5},
                {"x": 0.58, "y": 0.60, "w": 0.42, "h": 0.40, "z": 6}
            ]
        }
    ]
}

# =============================================================================
# ASPECT RATIO DEFINITIONS
# =============================================================================

ASPECT_RATIOS = {
    "1:1": 1.0,
    "16:9": 16.0 / 9.0,
    "9:16": 9.0 / 16.0,
    "3:2": 3.0 / 2.0,
    "2:3": 2.0 / 3.0,
    "4:3": 4.0 / 3.0,
    "3:4": 3.0 / 4.0
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_dimensions(aspect_ratio: float, canvas_total_pixels: int) -> tuple:
    """
    Calculate canvas dimensions for a given aspect ratio
    while maintaining the specified total pixel count.
    """
    target_megapixels = canvas_total_pixels * canvas_total_pixels
    height = int(math.sqrt(target_megapixels / aspect_ratio))
    width = int(height * aspect_ratio)
    return width, height


def calculate_fill_scale(img_width: int, img_height: int, region_width: int, region_height: int) -> float:
    """
    Calculate optimal scale to fill region while preserving aspect ratio.
    Scales to cover the region (may clip).
    """
    img_aspect = img_width / img_height
    region_aspect = region_width / region_height

    if img_aspect > region_aspect:
        # Image is wider - scale by height
        return region_height / img_height
    else:
        # Image is taller - scale by width
        return region_width / img_width


def score_layout(layout: dict, image_dimensions: list) -> float:
    """
    Score a layout based on how well images fit their regions.
    """
    total_score = 0.0

    for i, region in enumerate(layout["regions"]):
        if i >= len(image_dimensions):
            continue

        img_w, img_h = image_dimensions[i]
        img_aspect = img_w / img_h
        region_aspect = region["w"] / region["h"]

        # Score based on aspect ratio match (less clipping = better)
        aspect_diff = abs(img_aspect - region_aspect)
        aspect_score = 1.0 / (1.0 + aspect_diff)

        # Bonus for larger regions (prioritizes screen real estate)
        size_score = region["w"] * region["h"]

        total_score += aspect_score * 0.7 + size_score * 0.3

    return total_score


def apply_layout_rules(regions: list, image_count: int) -> list:
    """
    Apply the general layout rules:
    1. First image: top-left corner, largest at bottom z-order
    2. Last image: bottom-right corner, slightly larger with top z-order (spill effect)
    """
    if len(regions) < 2:
        return regions

    # Clone regions
    adjusted = [dict(r) for r in regions]

    # Find top-left region (smallest x+y)
    top_left_idx = 0
    min_top_left = adjusted[0]["x"] + adjusted[0]["y"]

    # Find bottom-right region (largest extent)
    bottom_right_idx = len(adjusted) - 1
    max_bottom_right = 0

    for i, r in enumerate(adjusted):
        top_left_score = r["x"] + r["y"]
        bottom_right_score = (r["x"] + r["w"]) + (r["y"] + r["h"])

        if top_left_score < min_top_left:
            min_top_left = top_left_score
            top_left_idx = i

        if bottom_right_score > max_bottom_right:
            max_bottom_right = bottom_right_score
            bottom_right_idx = i

    # Assign z-orders
    adjusted[top_left_idx]["z"] = 1
    adjusted[bottom_right_idx]["z"] = image_count

    z_counter = 2
    for i, r in enumerate(adjusted):
        if i != top_left_idx and i != bottom_right_idx:
            r["z"] = z_counter
            z_counter += 1

    return adjusted


def select_optimal_layout(image_count: int, image_dimensions: list) -> dict:
    """
    Select the best layout for given images based on aspect ratio matching.
    """
    count = min(max(image_count, 2), 6)
    strategies = LAYOUT_STRATEGIES.get(count, LAYOUT_STRATEGIES[6])

    # Find best scoring layout
    best_layout = strategies[0]
    best_score = -float("inf")

    for strategy in strategies:
        score = score_layout(strategy, image_dimensions)
        if score > best_score:
            best_score = score
            best_layout = strategy

    return {
        "type": best_layout["type"],
        "regions": apply_layout_rules(best_layout["regions"], count)
    }


def optimize_image_order(image_dimensions: list, regions: list) -> list:
    """
    Reorder images to best match regions based on aspect ratios.
    First image → top-left, Last image → bottom-right.
    """
    if len(image_dimensions) < 2:
        return list(range(len(image_dimensions)))

    # Find top-left and bottom-right regions
    top_left_region_idx = 0
    bottom_right_region_idx = len(regions) - 1
    min_top_left = regions[0]["x"] + regions[0]["y"]
    max_bottom_right = 0

    for i, r in enumerate(regions):
        top_left_score = r["x"] + r["y"]
        bottom_right_score = (r["x"] + r["w"]) + (r["y"] + r["h"])

        if top_left_score < min_top_left:
            min_top_left = top_left_score
            top_left_region_idx = i

        if bottom_right_score > max_bottom_right:
            max_bottom_right = bottom_right_score
            bottom_right_region_idx = i

    result = [-1] * len(regions)
    used_images = set()

    # Rule 1: First image (index 0) goes to top-left region
    result[top_left_region_idx] = 0
    used_images.add(0)

    # Rule 2: Last image goes to bottom-right region
    last_image_idx = len(image_dimensions) - 1
    result[bottom_right_region_idx] = last_image_idx
    used_images.add(last_image_idx)

    # For remaining regions, use aspect ratio matching
    remaining_regions = [
        (i, regions[i]) for i in range(len(regions))
        if i != top_left_region_idx and i != bottom_right_region_idx
    ]

    for region_idx, region in remaining_regions:
        best_img_idx = -1
        best_diff = float("inf")
        region_aspect = region["w"] / region["h"]

        for img_idx, (img_w, img_h) in enumerate(image_dimensions):
            if img_idx in used_images:
                continue
            img_aspect = img_w / img_h
            diff = abs(img_aspect - region_aspect)
            if diff < best_diff:
                best_diff = diff
                best_img_idx = img_idx

        if best_img_idx != -1:
            result[region_idx] = best_img_idx
            used_images.add(best_img_idx)

    return result


# =============================================================================
# COMFYUI NODE
# =============================================================================

class DJZ_Offsquare_V2:
    """
    Creates optimized image collages from batches of 2-6 images.
    First image appears top-left, last image appears bottom-right with spill effect.
    V2: Exposes canvas_total_pixels for customizable canvas size.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "aspect_ratio": (list(ASPECT_RATIOS.keys()), {"default": "1:1"}),
                "canvas_total_pixels": ("INT", {
                    "default": 3072,
                    "min": 512,
                    "max": 8192,
                    "step": 64,
                    "display": "number"
                }),
                "border_radius": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "display": "slider"
                }),
                "background_color": ("STRING", {"default": "#000000"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "layout_info")
    FUNCTION = "create_collage"
    CATEGORY = "image/compositing"

    def hex_to_rgb(self, hex_color: str) -> tuple:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def create_rounded_mask(self, width: int, height: int, radius: int) -> Image.Image:
        """Create a rounded rectangle mask."""
        mask = Image.new('L', (width, height), 0)
        if radius <= 0:
            mask.paste(255, (0, 0, width, height))
            return mask

        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)

        # Draw rounded rectangle
        draw.rounded_rectangle(
            [(0, 0), (width - 1, height - 1)],
            radius=radius,
            fill=255
        )
        return mask

    def create_collage(self, images: torch.Tensor, aspect_ratio: str,
                       canvas_total_pixels: int, border_radius: int,
                       background_color: str):
        """
        Main function to create the collage.

        Args:
            images: Batch of images as tensor (B, H, W, C)
            aspect_ratio: Selected aspect ratio string
            canvas_total_pixels: Base dimension for canvas size (e.g. 3072 = 3072x3072 at 1:1)
            border_radius: Corner radius for image regions
            background_color: Hex color for canvas background

        Returns:
            Tuple of (output_image_tensor, layout_info_string)
        """
        # Get batch size (number of images)
        batch_size = images.shape[0]

        if batch_size < 2:
            raise ValueError("OffSquare requires at least 2 images. Received: {}".format(batch_size))

        if batch_size > 6:
            print(f"[DJZ_Offsquare_V2] Warning: Maximum 6 images supported. Using first 6 of {batch_size}.")
            images = images[:6]
            batch_size = 6

        # Calculate canvas dimensions
        ratio = ASPECT_RATIOS[aspect_ratio]
        canvas_width, canvas_height = calculate_dimensions(ratio, canvas_total_pixels)

        # Convert tensors to PIL images and get dimensions
        pil_images = []
        image_dimensions = []

        for i in range(batch_size):
            # Convert from tensor (H, W, C) with values 0-1 to PIL
            img_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np, 'RGB')
            pil_images.append(pil_img)
            image_dimensions.append((pil_img.width, pil_img.height))

        # Select layout
        layout = select_optimal_layout(batch_size, image_dimensions)

        # Get optimized image order
        image_order = optimize_image_order(image_dimensions, layout["regions"])

        # Create canvas
        bg_color = self.hex_to_rgb(background_color)
        canvas = Image.new('RGB', (canvas_width, canvas_height), bg_color)

        # Sort regions by z-index for proper layering
        sorted_regions = sorted(
            enumerate(layout["regions"]),
            key=lambda x: x[1]["z"]
        )

        # Render each region
        for region_idx, region in sorted_regions:
            img_idx = image_order[region_idx]
            if img_idx < 0 or img_idx >= len(pil_images):
                continue

            pil_img = pil_images[img_idx]

            # Calculate region in pixels
            region_x = int(region["x"] * canvas_width)
            region_y = int(region["y"] * canvas_height)
            region_w = int(region["w"] * canvas_width)
            region_h = int(region["h"] * canvas_height)

            # Calculate scale to fill region
            scale = calculate_fill_scale(
                pil_img.width, pil_img.height,
                region_w, region_h
            )

            scaled_w = int(pil_img.width * scale)
            scaled_h = int(pil_img.height * scale)

            # Resize image
            scaled_img = pil_img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

            # Calculate offset to center in region
            offset_x = (region_w - scaled_w) // 2
            offset_y = (region_h - scaled_h) // 2

            # Create region image (crop to region size)
            region_img = Image.new('RGB', (region_w, region_h), bg_color)
            region_img.paste(scaled_img, (offset_x, offset_y))

            # Apply rounded corners if needed
            if border_radius > 0:
                mask = self.create_rounded_mask(region_w, region_h, border_radius)

                # Create a temporary canvas section
                temp_section = canvas.crop((region_x, region_y, region_x + region_w, region_y + region_h))

                # Composite with mask
                canvas.paste(region_img, (region_x, region_y), mask)
            else:
                canvas.paste(region_img, (region_x, region_y))

        # Convert back to tensor
        output_np = np.array(canvas).astype(np.float32) / 255.0
        output_tensor = torch.from_numpy(output_np).unsqueeze(0)  # Add batch dimension

        # Create layout info string
        layout_info = f"Layout: {layout['type']} | Images: {batch_size} | Canvas: {canvas_width}x{canvas_height} | Aspect: {aspect_ratio} | Base: {canvas_total_pixels}px"

        return (output_tensor, layout_info)


# =============================================================================
# NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "DJZ_Offsquare_V2": DJZ_Offsquare_V2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DJZ_Offsquare_V2": "DJZ Offsquare Collage V2"
}
