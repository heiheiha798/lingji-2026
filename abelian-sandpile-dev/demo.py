#!/usr/bin/env python3

import argparse
import collections
import math
import pathlib


def stabilize(size: int, grains: int):
    height = [0] * (size * size)
    odometer = [0] * (size * size)
    queued = bytearray(size * size)
    center = (size // 2) * size + size // 2
    height[center] = grains
    work = collections.deque([center])
    queued[center] = 1

    while work:
        index = work.popleft()
        queued[index] = 0
        q, height[index] = divmod(height[index], 4)
        if q == 0:
            continue
        odometer[index] += q
        y, x = divmod(index, size)
        neighbors = []
        if x:
            neighbors.append(index - 1)
        if x + 1 < size:
            neighbors.append(index + 1)
        if y:
            neighbors.append(index - size)
        if y + 1 < size:
            neighbors.append(index + size)
        for neighbor in neighbors:
            height[neighbor] += q
            if height[neighbor] >= 4 and not queued[neighbor]:
                queued[neighbor] = 1
                work.append(neighbor)
    return height, odometer


def write_ppm(path: pathlib.Path, size: int, pixels):
    header = f"P6\n{size} {size}\n255\n".encode("ascii")
    path.write_bytes(header + bytes(pixels))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a small Abelian sandpile")
    parser.add_argument("--size", type=int, default=257)
    parser.add_argument("--grains", type=int, default=1_000_000)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("demo"))
    args = parser.parse_args()
    if args.size < 3 or args.grains < 0:
        parser.error("size must be >= 3 and grains must be non-negative")

    stable, odometer = stabilize(args.size, args.grains)
    stable_palette = [(15, 23, 42), (37, 99, 235), (34, 197, 94), (250, 204, 21)]
    stable_pixels = [channel for value in stable for channel in stable_palette[value]]
    peak = max(odometer, default=0)
    scale = math.log1p(peak) or 1.0
    odo_pixels = []
    for value in odometer:
        t = math.log1p(value) / scale
        odo_pixels.extend((int(255 * t), int(180 * t * t), int(255 * (1 - t))))

    stable_path = args.output.with_name(args.output.name + "_stable.ppm")
    odometer_path = args.output.with_name(args.output.name + "_odometer.ppm")
    write_ppm(stable_path, args.size, stable_pixels)
    write_ppm(odometer_path, args.size, odo_pixels)
    print(stable_path)
    print(odometer_path)


if __name__ == "__main__":
    main()

