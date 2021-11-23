from math import floor
import numpy as np


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


def ascii_display(pixels):
    for line in pixels:
        print(join_array(line))


def main():
    res = "400, 230"  # input("resolution(width, height):")
    w, h = [int(x) for x in res.split(",")]
    y_vals = np.array([(y + 1) / (h + 1) for y in range(h)])
    c_vals = np.array([x / w for x in range(w)]) * 3 + 1
    results = np.zeros((w, h))
    for i, c in enumerate(c_vals):
        results[i] = log_map_rep(c, y_vals, 500)
    pixels = np.zeros((w, h), dtype=np.int8)
    for x, column in enumerate(results):
        for y in y_val_to_index(column, h):
            pixels[x][int(y)] = 1
    return pixels


if __name__ == "__main__":
    pixels = main()
