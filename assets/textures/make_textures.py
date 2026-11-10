"""Writes cardboard.png and concrete.png: 512 px procedural textures for
props, obstacles and the floor. Seeded, so reruns give the same files.

Run: uv run python assets/textures/make_textures.py
"""

from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 512
HERE = Path(__file__).parent


def blotches(rng: np.random.Generator, cells: int) -> np.ndarray:
    """Smooth noise in [0, 1] that tiles without seams: a coarse random grid,
    repeated 3 x 3, scaled up bicubically, middle tile kept."""
    coarse = np.tile((rng.random((cells, cells)) * 255).astype(np.uint8), (3, 3))
    big = Image.fromarray(coarse).resize((3 * SIZE, 3 * SIZE), Image.BICUBIC)
    return np.asarray(big, float)[SIZE : 2 * SIZE, SIZE : 2 * SIZE] / 255.0


def save(name: str, rgb: np.ndarray) -> None:
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(HERE / name)


def main() -> None:
    rng = np.random.default_rng(7)
    # cardboard: kraft brown, soft blotches, faint flutes along x, fine grain
    shade = 0.85 + 0.15 * blotches(rng, 8)
    flutes = 1.0 + 0.04 * np.sin(np.arange(SIZE) / SIZE * 2 * np.pi * 48)[None, :]
    grain = 1.0 + 0.05 * (rng.random((SIZE, SIZE)) - 0.5)
    tone = shade * flutes * grain
    save("cardboard.png", np.stack([196 * tone, 150 * tone, 98 * tone], axis=-1))
    # concrete: mid grey, large stains, speckle
    stains = 0.8 + 0.2 * blotches(rng, 6)
    speckle = 1.0 + 0.18 * (rng.random((SIZE, SIZE)) - 0.5)
    grey = 150 * stains * speckle
    save("concrete.png", np.stack([grey, grey, grey * 1.02], axis=-1))


if __name__ == "__main__":
    main()
