"""
Crop the balloons from https://opengameart.org/content/balloon-pop-sprite.

Make sure you downloaded them before to ``popthatballoon/media/sprites``.
"""

import argparse
import os
import pathlib
import sys
import tempfile

import PIL
import PIL.Image


def main() -> int:
    """Execute the main routine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    this_path = pathlib.Path(os.path.realpath(__file__))
    sprites_dir = this_path.parent.parent / "popthatballoon/media/sprites"

    image = PIL.Image.open(str(sprites_dir / "balloon_red.png"))
    png_info = dict()
    if image.mode not in ['RGB', 'RGBA']:
        image = image.convert('RGBA')
        png_info = image.info

    image_w, image_h = image.size
    sprite_w = int(image_w / 3)
    sprite_h = int(image_h / 2)

    for i in range(6):
        xmin = (i % 3) * sprite_w
        xmax = (i % 3) * sprite_w + sprite_w

        ymin = int(i / 3) * sprite_h
        ymax = int(i / 3) * sprite_h + sprite_h

        cropped = image.crop((xmin, ymin, xmax, ymax))
        cropped = cropped.resize((50, 100))

        if i < 3:
            cropped.save(str(sprites_dir / f"balloon_idling{i}.png"), **png_info)
        else:
            cropped.save(str(sprites_dir / f"balloon_popping{i - 3}.png"), **png_info)

    return 0


if __name__ == "__main__":
    sys.exit(main())
