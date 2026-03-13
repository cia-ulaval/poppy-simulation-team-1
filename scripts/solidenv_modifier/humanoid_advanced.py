import xml.etree.ElementTree as ET
import random
import uuid

source= "humanoid.xml"


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