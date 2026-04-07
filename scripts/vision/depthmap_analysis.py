import numpy as np
import pathlib

source=pathlib.Path("./scripts/vision/frames")
Frames=[np.load(frame) for frame in source.iterdir()]

def find_divisors_generator(n):
    for i in range(1, int(n**0.5) + 1):  # Loop up to √n
        if n % i == 0:
            yield i
            if i != n // i:
                yield n // i

def frame_splitting(frame, n_regions, transpose=True):
    newframe=[]
    region=[]
    if transpose:
        frame=np.transpose(frame)
    size=frame.shape[0]
    possible_nregions=list(find_divisors_generator(size))
    if n_regions not in possible_nregions:
        print(f'{n_regions} does not divide cleanly, bumping up to closest divisor')
        possible_nregions.append(n_regions)
        possible_nregions_array=np.array(possible_nregions)
        sort_possibilites=np.sort(possible_nregions_array)
        index=np.where(sort_possibilites==n_regions)[0][0]
        n_regions=sort_possibilites[index+1]
        
    regionsize=size//n_regions
    lengths=[]
    for i, section in enumerate(frame):
        lengths.append(section.shape)
        if i > 0 and i % regionsize == 0:
            newframe.append(np.array(region).flatten())
            region = []
        region.append(section)

    if region:
        newframe.append(np.array(region).flatten())
    return np.array(newframe)

def splitframe_to_1Ddepthmap(frame, focal):
    dmap=[]
    for region in frame:
        Region=region.tolist()
        dmap.append((focal*min(Region)/300))
    return dmap

def loop(frames, n_region, focal):
    maps=[]
    splitFrames=[]
    for Frame in frames:
        splitframe=frame_splitting(Frame, n_region)
        splitFrames.append(splitframe)
        dmap=splitframe_to_1Ddepthmap(splitframe, focal)
        maps.append(dmap)
    return maps


# splitframe=frame_splitting(Frames[0], 11)
# print(splitframe.shape[0]*splitframe.shape[1], splitframe.shape, Frames[0].shape[0]*Frames[0].shape[1], Frames[0].shape)
# regionmap=splitframe_to_1Ddepthmap(splitframe, 300)
# print(regionmap, len(regionmap))
