import pathlib

import numpy as np

source = pathlib.Path("./scripts/vision/frames")
Frames = [np.load(frame) for frame in source.iterdir()]


def find_divisors_generator(n):
    for i in range(1, int(n**0.5) + 1):  # Loop up to √n
        if n % i == 0:
            yield i
            if i != n // i:
                yield n // i


def frame_splitting(frame, n_regions, transpose=True):
    newframe = []
    region = []
    if transpose:
        frame = np.transpose(frame)
    size = frame.shape[0]
    possible_nregions = list(find_divisors_generator(size))
    if n_regions not in possible_nregions:
        print(f"{n_regions} does not divide cleanly, bumping up to closest divisor")
        possible_nregions.append(n_regions)
        possible_nregions_array = np.array(possible_nregions)
        sort_possibilites = np.sort(possible_nregions_array)
        index = np.where(sort_possibilites == n_regions)[0][0]
        n_regions = sort_possibilites[index + 1]

    regionsize = size // n_regions
    lengths = []
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
    dmap = []
    for region in frame:
        Region = region.tolist()
        dmap.append((focal * min(Region) / 300))
    return dmap


def close_warning(dmap, threshold, frame, focal):
    warning_map = []
    region = []
    size = frame.size
    n_regions = len(dmap)
    length = size[0]
    lregion = length / n_regions
    regions = [0 + l * lregion for l in range(n_regions)]
    for i, distance in enumerate(dmap):
        if distance < threshold:
            for pixel in splitframe[i]:
                pixel = (300 * distance) / focal
                region.append(pixel)
            warning_map.append(region)
        else:
            warning_map.append(splitframe[i])
    return np.array(warning_map)


def loop(frames, n_region, focal):
    maps = []
    splitFrames = []
    for Frame in frames:
        splitframe = frame_splitting(Frame, n_region)
        splitFrames.append(splitframe)
        dmap = splitframe_to_1Ddepthmap(splitframe, focal)
        maps.append(dmap)
    return maps


if __name__ == "__main__":
    splitframe = frame_splitting(Frames[0], 10)
    # print(splitframe.shape[0]*splitframe.shape[1], splitframe.shape, Frames[0].shape[0]*Frames[0].shape[1], Frames[0].shape)
    regionmap = splitframe_to_1Ddepthmap(splitframe, 3.2)
    # print(regionmap, len(regionmap))
    warningmap = close_warning(regionmap, 0.2, splitframe, 3.2)
    print(
        warningmap,
        warningmap.shape,
        splitframe.shape,
        Frames[0].shape[0] * Frames[0].shape[1],
    )


def make_regions(frame, nb_regions):
    region_width = frame.shape[1] // nb_regions
    return [
        frame[:, i * region_width : (i + 1) * region_width] for i in range(nb_regions)
    ]


def regions_depth(frame, nb_regions):
    regions = make_regions(frame, nb_regions)
    return np.array([np.min(region) for region in regions])
