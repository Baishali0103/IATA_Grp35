import os
import csv
import random
from dataclasses import dataclass, asdict
from typing import Tuple, List

from PIL import Image, ImageDraw

IMAGE_SIZE = 128
NUM_SAMPLES = 8000
OUTPUT_DIR = "output"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
CSV_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")

random.seed(42)

COLORS = {
    "red": (220, 50, 50),
    "blue": (50, 90, 220),
    "green": (50, 170, 90),
    "yellow": (230, 200, 40),
}

SHAPES = ["circle", "square", "triangle"]
SIZES = ["small", "big"]
RELATIONS = ["above", "below", "left of", "right of"]

SIZE_TO_PIXELS = {
    "small": 18,
    "big": 30,
}



@dataclass
class Sample:
    sample_id: int
    obj1_size: str
    obj1_color: str
    obj1_shape: str
    relation: str
    obj2_size: str
    obj2_color: str
    obj2_shape: str
    sentence: str

def ensure_dirs() -> None:
    os.makedirs(IMAGE_DIR, exist_ok=True)


def random_choice_excluding(options: List[str], exclude: str) -> str:
    candidates = [x for x in options if x != exclude]
    return random.choice(candidates)


def generate_semantic_sample(sample_id: int) -> dict:

    obj1_size = random.choice(SIZES)
    obj1_color = random.choice(list(COLORS.keys()))
    obj1_shape = random.choice(SHAPES)

    obj2_size = random.choice(SIZES)
    obj2_color = random.choice(list(COLORS.keys()))
    obj2_shape = random.choice(SHAPES)

    relation = random.choice(RELATIONS)

    if (
        obj1_size == obj2_size
        and obj1_color == obj2_color
        and obj1_shape == obj2_shape
    ):
        obj2_color = random_choice_excluding(list(COLORS.keys()), obj1_color)

    return {
        "sample_id": sample_id,
        "obj1_size": obj1_size,
        "obj1_color": obj1_color,
        "obj1_shape": obj1_shape,
        "relation": relation,
        "obj2_size": obj2_size,
        "obj2_color": obj2_color,
        "obj2_shape": obj2_shape,
    }


def render_sentence(sample: dict) -> str:

    return (
        f"a {sample['obj1_size']} {sample['obj1_color']} {sample['obj1_shape']} "
        f"is {sample['relation']} "
        f"a {sample['obj2_size']} {sample['obj2_color']} {sample['obj2_shape']}"
    )


def sample_positions(relation: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:

    jitter = lambda low, high: random.randint(low, high)

    if relation == "above":
        obj1 = (jitter(35, 93), jitter(20, 45))
        obj2 = (jitter(35, 93), jitter(80, 108))
    elif relation == "below":
        obj1 = (jitter(35, 93), jitter(80, 108))
        obj2 = (jitter(35, 93), jitter(20, 45))
    elif relation == "left of":
        obj1 = (jitter(20, 45), jitter(35, 93))
        obj2 = (jitter(80, 108), jitter(35, 93))
    elif relation == "right of":
        obj1 = (jitter(80, 108), jitter(35, 93))
        obj2 = (jitter(20, 45), jitter(35, 93))
    else:
        raise ValueError(f"Unsupported relation: {relation}")

    return obj1, obj2


def draw_circle(draw: ImageDraw.ImageDraw, center: Tuple[int, int], radius: int, color: Tuple[int, int, int]) -> None:
    x, y = center
    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bbox, fill=color, outline=(0, 0, 0), width=2)


def draw_square(draw: ImageDraw.ImageDraw, center: Tuple[int, int], half_size: int, color: Tuple[int, int, int]) -> None:
    x, y = center
    bbox = [x - half_size, y - half_size, x + half_size, y + half_size]
    draw.rectangle(bbox, fill=color, outline=(0, 0, 0), width=2)


def draw_triangle(draw: ImageDraw.ImageDraw, center: Tuple[int, int], size: int, color: Tuple[int, int, int]) -> None:
    x, y = center
    points = [
        (x, y - size),
        (x - size, y + size),
        (x + size, y + size),
    ]
    draw.polygon(points, fill=color, outline=(0, 0, 0))


def draw_shape(
    draw: ImageDraw.ImageDraw,
    shape: str,
    center: Tuple[int, int],
    size_label: str,
    color_name: str
) -> None:
    size_px = SIZE_TO_PIXELS[size_label]
    color = COLORS[color_name]

    if shape == "circle":
        draw_circle(draw, center, size_px, color)
    elif shape == "square":
        draw_square(draw, center, size_px, color)
    elif shape == "triangle":
        draw_triangle(draw, center, size_px, color)
    else:
        raise ValueError(f"Unsupported shape: {shape}")


def render_image(sample: dict, save_path: str) -> None:

    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    obj1_center, obj2_center = sample_positions(sample["relation"])

    draw_shape(
        draw=draw,
        shape=sample["obj1_shape"],
        center=obj1_center,
        size_label=sample["obj1_size"],
        color_name=sample["obj1_color"],
    )

    draw_shape(
        draw=draw,
        shape=sample["obj2_shape"],
        center=obj2_center,
        size_label=sample["obj2_size"],
        color_name=sample["obj2_color"],
    )

    image.save(save_path)


def save_metadata(samples: List[Sample]) -> None:
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "obj1_size",
                "obj1_color",
                "obj1_shape",
                "relation",
                "obj2_size",
                "obj2_color",
                "obj2_shape",
                "sentence",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))


def main() -> None:
    ensure_dirs()
    results: List[Sample] = []

    for i in range(NUM_SAMPLES):
        semantic_sample = generate_semantic_sample(i)
        sentence = render_sentence(semantic_sample)
        semantic_sample["sentence"] = sentence

        image_path = os.path.join(IMAGE_DIR, f"sample_{i:02d}.png")
        render_image(semantic_sample, image_path)

        sample_obj = Sample(**semantic_sample)
        results.append(sample_obj)

        print(f"[{i}] {sentence} -> {image_path}")

    save_metadata(results)
    print(f"\nSaved metadata to: {CSV_PATH}")
    print(f"Saved images to: {IMAGE_DIR}")


if __name__ == "__main__":
    main()