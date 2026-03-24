import xml.etree.ElementTree as ET
import random
import uuid
import numpy as np
from PIL import Image

source= "humanoid.xml"

def duplicate_env(source):
    tree = ET.parse(source)

    uid = str(uuid.uuid4())
    new_file = f'humanoid_{uid}.xml'

    tree.write(new_file)

    return new_file

def make_obstacle():
    x=random.uniform(10,30)
    y=random.uniform(10,30)
    geom_types=["sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh"]
    gtype=geom_types[random.randint(0, len(geom_types)-1)]
    xsize=random.uniform(0,10)
    ysize=random.uniform(0,10)
    zsize=random.uniform(0,10)
    if gtype=="sphere":
        size=str(zsize)
    if gtype in ("capsule", "cylinder"):
        size=f' {xsize} {ysize}'
    else:
        size=f' {xsize} {ysize} {zsize}'
    red=random.uniform(0,1)
    green=random.uniform(0,1)
    blue=random.uniform(0,1)
    rgba=f'{red} {green} {blue} 1'
    pos=f'{x} {y} 0'
    density=str(random.randint(1,10000))
    return pos, gtype, size, rgba, density

def change_floor(source):
    tree=ET.parse(source)
    Root=tree.getroot()
   # -------- generate noise heightfield --------
    size = 256

    noise = np.random.rand(size, size)

    # simple smoothing to imitate Perlin-like terrain
    for _ in range(5):
        noise = (
            noise +
            np.roll(noise, 1, 0) +
            np.roll(noise, -1, 0) +
            np.roll(noise, 1, 1) +
            np.roll(noise, -1, 1)
        ) / 5.0

    # normalize
    noise = (noise - noise.min()) / (noise.max() - noise.min())

    # convert to image (MuJoCo reads grayscale heightfields)
    img = (noise * 255).astype(np.uint8)
    Image.fromarray(img).save("terrain.png")

    # -------- add hfield asset --------
    asset = Root.find("asset")

    hfield = ET.SubElement(asset, "hfield")
    hfield.set("name", "terrain")
    hfield.set("file", "terrain.png")

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

    tree.write('humanoid.xml')

def restore_floor(source):
    tree=ET.parse(source)
    root=tree.getroot()
    for asset in root.findall('asset'):
        for hfield in asset.findall('hfield'):
            if hfield.get('name') == 'terrain':
                asset.remove(hfield)
    for world in root.findall('worldbody'):
        for geom in world.findall('geom'):
            if geom.get('name') == 'floor':
                geom.set('type', 'plane')
                geom.set('size', '20 20 0.125')
                if 'hfield' in geom.attrib:
                    del geom.attrib['hfield']

    tree.write(source)
    

def add_obastacles(n_obstacles=10 ,source=source):
    Tree=ET.parse(source)
    root=Tree.getroot()
    for n in range(n_obstacles):
        pos, gtype, size, rgba, density = make_obstacle()
        name='obstacle'
        if gtype=="mesh":
            n_vertex=random.randint[3, 20]
            scale=size
        for worldbody in root.findall('worldbody'):
            body = ET.SubElement(worldbody, 'body')
            body.set('name', name)
            body.set('pos', pos)
            geom = ET.SubElement(body, "geom")
            geom.set('type', gtype)
            geom.set('size', size)
            geom.set('rgba', rgba)
            geom.set('density', density)
    Tree.write('humanoid.xml')

def remove_obstacles(source):
    tree=ET.parse(source)
    Root=tree.getroot()
    for world in Root.findall('worldbody'):
        for body in world.findall('body'):
            name = body.get('name')
            if name == 'obstacle':
                world.remove(body)

    tree.write('humanoid.xml')