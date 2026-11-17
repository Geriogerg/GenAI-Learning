"""
cv_tools + the VLM, wrapped for agents: images are referred to by short names ("plate_012.jpg",
"crops/plate_012_80_130_200_260.jpg") instead of full paths, outputs stay small, and the
docstrings say when to use each tool. Used by 04, 06 and the MCP server in 07.
"""
from pathlib import Path
from typing import Annotated

from pydantic import Field

import cv_tools as cv
from vision_utils import OUTPUT_DIR, PLATES_DIR, REFERENCE, Agent, tool, vlm  # noqa: F401

ImageName = Annotated[str, Field(description="Image name as returned by list_images, or a crop path from crop_region")]


def resolve(name):
    for candidate in (Path(name), PLATES_DIR / name, OUTPUT_DIR / name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"image '{name}' not found. Use list_images to see the available names.")


@tool
def list_images():
    """List the images of the current batch (reference image excluded)."""
    return sorted(p.name for p in PLATES_DIR.glob("*.jpg") if p.name != REFERENCE.name)


@tool
def check_image_quality(image: ImageName):
    """Check sharpness and exposure. Do this first: results on an unusable image are unreliable."""
    return cv.check_image_quality(resolve(image))


@tool
def find_anomalies(image: ImageName):
    """Compare the image with the golden sample (known-good plate) and return the regions that differ,
    as bounding boxes [x1, y1, x2, y2]. Very sensitive, says WHERE something is unusual, not WHAT it is."""
    return cv.find_anomalies(resolve(image))


@tool
def crop_region(image: ImageName, x1: int, y1: int, x2: int, y2: int):
    """Cut and enlarge a region (e.g. a bbox from find_anomalies). Returns a crop name that can be passed
    to ask_vlm for a close-up look."""
    res = cv.crop_region(resolve(image), x1, y1, x2, y2)
    return {"crop": str(Path(res["crop_path"]).relative_to(OUTPUT_DIR)).replace("\\", "/"), "region": res["region"]}


@tool
def ask_vlm(image: ImageName, question: str, with_reference: bool = False):
    """Ask a vision-language model a question about an image or a crop. Good at naming what it sees
    (type of defect, its look), weak at tiny details on a full image: crop first when you can.
    with_reference=True also shows it the golden sample (only makes sense for full plate images)."""
    images = [REFERENCE, resolve(image)] if with_reference else [resolve(image)]
    prefix = "The FIRST image is a known-good reference, the SECOND is the one to look at. " if with_reference else ""
    return vlm(images, prefix + question, max_tokens=1000)


@tool
def detect_objects(image: ImageName):
    """Detect everyday objects (80 COCO classes: people, vehicles, tools...) with YOLO.
    Useless for surface defects."""
    return cv.detect_objects(resolve(image))


@tool
def clip_classify(image: ImageName, labels: list[str]):
    """Zero-shot CLIP scores for text labels on the whole image. Coarse: misses small local details."""
    return cv.clip_classify(resolve(image), labels)


TOOLS = [list_images, check_image_quality, find_anomalies, crop_region, ask_vlm, detect_objects, clip_classify]
