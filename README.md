# ComfyUI-DJZ-Offsquare

A ComfyUI custom node that creates optimized image collages from batches of 2-6 images with intelligent layout selection and aspect ratio control.

## Features

- **Multiple Layout Strategies** — 3 layouts per image count (2-6 images)
- **Aspect Ratio Selection** — 1:1, 16:9, 9:16, 3:2, 2:3, 4:3, 3:4
- **Consistent Output Size** — Maintains ~4 megapixels (same as 2048×2048)
- **Smart Image Placement** — First image → top-left, Last image → bottom-right with spill effect
- **Aspect-Ratio Aware** — Matches images to regions for minimal clipping
- **Auto Layout** — Automatically selects best layout based on image dimensions

## Installation

### Option 1: Git Clone (Recommended)

1. Navigate to your ComfyUI custom nodes folder:
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/MushroomFleet/ComfyUI-DJZ-Offsquare.git
   ```

3. Restart ComfyUI

### Option 2: Manual Installation

1. Download the repository as a ZIP file
2. Extract to `ComfyUI/custom_nodes/ComfyUI-DJZ-Offsquare/`
3. Restart ComfyUI

The node will appear under: **image/compositing → DJZ Offsquare Collage**

## Inputs

| Input | Type | Description |
|-------|------|-------------|
| `images` | IMAGE | Batch of 2-6 images |
| `aspect_ratio` | Dropdown | Output aspect ratio (default: 1:1) |
| `layout_style` | Dropdown | Layout strategy or "auto" |
| `border_radius` | INT (0-64) | Corner radius for image regions |
| `background_color` | STRING | Hex color for canvas background |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | The generated collage |
| `layout_info` | STRING | Layout details (type, dimensions, etc.) |

## Layout Styles

### 2 Images
- `diagonal` — Overlapping diagonal composition
- `vsplit` — Vertical split
- `hsplit` — Horizontal split

### 3 Images
- `cascade` — Full-width top + two bottom
- `l-shape` — Large corner + two edges
- `reverse-l` — Tall left column + two right

### 4 Images
- `grid` — 2×2 with bottom-right emphasis
- `dominant-three` — Large left + three stacked right
- `t-layout` — Wide top + three bottom

### 5 Images
- `quincunx` — Four corners + small center
- `asymmetric` — Varied sizes with center accent
- `l-shape` — Large corner + four edges

### 6 Images
- `grid-2x3` — 2×3 grid with spill
- `grid-3x2` — 3×2 grid with spill
- `feature-corner` — Large corner + five tiles

## Aspect Ratios & Dimensions

All ratios maintain ~4,194,304 total pixels:

| Ratio | Dimensions | Use Case |
|-------|------------|----------|
| 1:1 | 2048 × 2048 | Square |
| 16:9 | 2731 × 1536 | Widescreen |
| 9:16 | 1536 × 2731 | Vertical/Mobile |
| 3:2 | 2509 × 1672 | Classic Photo |
| 2:3 | 1672 × 2509 | Portrait Photo |
| 4:3 | 2365 × 1774 | Standard |
| 3:4 | 1774 × 2365 | Portrait Standard |

## Usage Tips

1. **Image Order Matters** — First image in batch goes to top-left (background), last image goes to bottom-right (foreground with spill)

2. **Use Auto Layout** — Let the algorithm pick the best layout based on your images' aspect ratios

3. **Batch Creation** — Use a "Batch Images" node to combine individual images before connecting to this node

4. **Chain with Save** — Connect output directly to "Save Image" node for export

## Example Workflow

```
[Load Image 1] ─┐
[Load Image 2] ─┼─► [Batch Images] ─► [DJZ Offsquare] ─► [Save Image]
[Load Image 3] ─┤
[Load Image 4] ─┘
```

## Requirements

- ComfyUI (latest version recommended)
- Python 3.8+
- PIL/Pillow
- PyTorch
- NumPy

## License

MIT License

## Citation

```bibtex
@software{djz_offsquare,
  title = {DJZ-Offsquare: Intelligent Image Collage Component},
  author = {Drift Johnson},
  year = {2025},
  url = {https://github.com/MushroomFleet/ComfyUI-DJZ-Offsquare},
  version = {1.0.0}
}
```

## Support

[![Ko-Fi](https://cdn.ko-fi.com/cdn/kofi3.png?v=3)](https://ko-fi.com/driftjohnson)
