from math import floor
import numpy as np
import struct
import zlib


def lmap(c: float, x: float) -> float:  # logistic map
    return c * x * (1 - x)


def log_map_rep(c: float, x: float, rep: int) -> float:
    for i in range(rep):
        x = lmap(c, x)
    return x


def log_map_tol(c: float, xi: float, tol: int) -> float:
    xs = np.array([xi, lmap(c, xi)])
    repeat = False
    while not repeat:
        xs = np.append(xs, lmap(c, xs[-1]))
        repeat = any(abs(xs[:-1] - xs[-1]) < tol)
    return xs[-1]


def y_val_to_index(y: float, h: int) -> int:
    return np.floor(y * h)


def join_array(array):
    return "".join(["O" if x else "-" for x in list(array)])


def ascii_display(pixels) -> None:
    for line in pixels:
        print(join_array(line))


def pixels_to_png(pixels, w: int, h: int) -> None:
    png = b"\x89PNG\r\n\x1A\n"
    png += struct.pack(">I", 13)
    IHDR = b"IHDR"
    IHDR += struct.pack(">I", w - 1)
    IHDR += struct.pack(">I", h - 1)
    IHDR += struct.pack(">B", 8)  # bit_depth
    IHDR += struct.pack(">B", 0)  # color_type
    IHDR += struct.pack(">B", 0)  # compression_method
    IHDR += struct.pack(">B", 0)  # filter_method
    IHDR += struct.pack(">B", 0)  # interlace_method
    png += IHDR
    png += struct.pack(">I", zlib.crc32(IHDR))
    raw = b""
    for y in range(h - 1, 0, -1):
        for x in range(w):
            p = 255 if pixels[x][y] else 0
            raw += struct.pack(">B", p)
    compressor = zlib.compressobj()
    compressed = compressor.compress(raw)
    compressed += compressor.flush()
    png += struct.pack(">I", len(compressed))
    png += b"IDAT" + compressed
    png += struct.pack(">I", zlib.crc32(b"IDAT" + compressed))
    png += struct.pack(">I", 0)
    png += b"IEND" + struct.pack(">I", zlib.crc32(b"IEND"))
    with open("logistic.png", "wb") as image:
        image.write(png)


def main() -> None:
    res = input("resolution(width, height):")
    itterations = input("desired number of itterations:")
    w, h = [int(x) for x in res.split(",")]
    y_vals = np.array([(y + 1) / (h + 1) for y in range(h)])
    c_vals = np.array([x / w for x in range(w)]) * 3 + 1
    results = np.zeros((w, h))
    for i, c in enumerate(c_vals):
        results[i] = log_map_rep(c, y_vals, 1000)
    pixels = np.zeros((w, h), dtype=np.int8)
    for x, column in enumerate(results):
        for y in y_val_to_index(column, h):
            pixels[x][int(y)] = 1
    pixels_to_png(pixels, w, h)


if __name__ == "__main__":
    main()
