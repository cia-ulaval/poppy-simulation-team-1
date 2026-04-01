import os
import random
import tempfile
import uuid
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

source = "humanoid.xml"


def duplicate_env(source, output_dir=None):
    tree = ET.parse(source)

    uid = str(uuid.uuid4())
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    new_file = os.path.join(output_dir, f"humanoid_{uid}.xml")
    tree.write(new_file)

    return new_file


def make_obstacle():
    x = random.uniform(-10, 10)
    y = random.uniform(-10, 10)
    radius = random.uniform(0.01, 1)
    return x, y, radius


def add_obstacle(x, y, radius, source, i):
    Tree = ET.parse(source)
    root = Tree.getroot()
    name = f"obstacle_{i}"
    for worldbody in root.findall("worldbody"):
        body = ET.SubElement(worldbody, "body")
        body.set("name", name)
        body.set("pos", f"{x} {y} 2.5")
        geom = ET.SubElement(body, "geom")
        geom.set("type", "cylinder")
        geom.set("size", f"{radius} 5")
        geom.set("rgba", "1 1 1 1")
        geom.set("density", "1000")
    Tree.write(source)


def add_obstacles(n_obstacles=10, source=source):
    for i in range(n_obstacles):
        x, y, radius = make_obstacle()
        while abs(x) - radius < 0.2 or abs(y) - radius < 0.2:
            x, y, radius = make_obstacle()
        add_obstacle(x, y, radius, source, i)


def remove_obstacles(source):
    tree = ET.parse(source)
    Root = tree.getroot()
    for world in Root.findall("worldbody"):
        for body in world.findall("body"):
            name = body.get("name")
            if name == "obstacle":
                world.remove(body)

    tree.write(source)


def change_floor(source, output_dir=None):
    tree = ET.parse(source)
    Root = tree.getroot()
    # -------- generate noise heightfield --------
    size = 256

    noise = np.random.rand(size, size)

    # simple smoothing to imitate Perlin-like terrain
    for _ in range(5):
        noise = (
            noise
            + np.roll(noise, 1, 0)
            + np.roll(noise, -1, 0)
            + np.roll(noise, 1, 1)
            + np.roll(noise, -1, 1)
        ) / 5.0

    # normalize
    noise = (noise - noise.min()) / (noise.max() - noise.min())

    # convert to image (MuJoCo reads grayscale heightfields)
    img = (noise * 255).astype(np.uint8)
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    terrain_path = os.path.join(output_dir, "terrain.png")
    Image.fromarray(img).save(terrain_path)

    # -------- add hfield asset --------
    asset = Root.find("asset")

    hfield = ET.SubElement(asset, "hfield")
    hfield.set("name", "terrain")
    hfield.set("file", terrain_path)

    # size = half-length-x, half-length-y, max-height, base-height
    hfield.set("size", "20 20 2 0.1")

    # -------- replace plane floor --------
    for world in Root.findall("worldbody"):
        for geom in world.findall("geom"):
            if geom.get("name") == "floor":
                geom.set("type", "hfield")
                geom.set("hfield", "terrain")

                # plane-specific size not needed anymore
                if "size" in geom.attrib:
                    del geom.attrib["size"]

    tree.write(source)


def restore_floor(source):
    tree = ET.parse(source)
    root = tree.getroot()
    for asset in root.findall("asset"):
        for hfield in asset.findall("hfield"):
            if hfield.get("name") == "terrain":
                asset.remove(hfield)
    for world in root.findall("worldbody"):
        for geom in world.findall("geom"):
            if geom.get("name") == "floor":
                geom.set("type", "plane")
                geom.set("size", "20 20 0.125")
                if "hfield" in geom.attrib:
                    del geom.attrib["hfield"]

    tree.write(source)
