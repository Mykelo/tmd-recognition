from math import ceil
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import pickle


def draw_points(
    img: np.ndarray,
    points: np.ndarray,
    scale: int = 1,
    target_width: int = 800,
    save_path: str or None = None,
    draw_indexes: bool = False,
):
    width_scale = target_width / img.shape[0]

    test_img = cv2.resize(
        img.copy(),
        (
            int(img.shape[1] * width_scale * scale),
            int(img.shape[0] * width_scale * scale),
        ),
    )
    for i in range(points.shape[0] // 2):
        if points.max() <= 1:
            x_pred = points[i * 2] * test_img.shape[1]
            y_pred = points[i * 2 + 1] * test_img.shape[0]
        else:
            x_pred = points[i * 2] * width_scale * scale
            y_pred = points[i * 2 + 1] * width_scale * scale
        cv2.circle(
            test_img,
            (int(x_pred), int(y_pred)),
            ceil(test_img.shape[0] / 250),
            (0, 255, 0),
            -1,
        )
        if draw_indexes:
            cv2.putText(
                test_img,
                f"{i}",
                (int(x_pred) + 4, int(y_pred) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
            )

    rgb_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    f, a = plt.subplots(1, 1, figsize=(12, 12))
    a.axis("off")
    a.imshow(rgb_img)
    if save_path is not None:
        cv2.imwrite(save_path, test_img)


def draw_line_through_points(
    img: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> np.ndarray:
    height, width, _ = img.shape
    dims = np.array([width, height])
    p1 = (p1 * dims).astype(np.int32)
    p2 = (p2 * dims).astype(np.int32)

    if p1[0] == p2[0]:
        draw_p1 = (p1[0], 0)
        draw_p2 = (p2[0], height)
    else:
        points = [p1, p2]
        x_coords, y_coords = zip(*points)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        a, b = np.linalg.lstsq(A, y_coords, rcond=None)[0]
        y0 = int(b)
        ymax = int(a * width + b)
        draw_p1, draw_p2 = (0, y0), (width, ymax)

    cv2.line(img, draw_p1, draw_p2, (255, 0, 0), 2)


def draw_point(img: np.ndarray, point: np.ndarray, radius, color, thickness):
    height, width, _ = img.shape
    dims = np.array([width, height])
    point = (point * dims).astype(np.int32)

    cv2.circle(img, point, radius, color, thickness)


def draw_line(img: np.ndarray, p1: np.ndarray, p2: np.ndarray, color, thickness):
    height, width, _ = img.shape
    dims = np.array([width, height])
    p1 = (p1 * dims).astype(np.int32)
    p2 = (p2 * dims).astype(np.int32)

    return cv2.line(img, p1, p2, color, thickness)


def draw_polygon(img: np.ndarray, pts: np.ndarray, color):
    height, width, _ = img.shape
    dims = np.array([width, height])
    pts = pts.copy()
    pts[:, 0] *= width
    pts[:, 1] *= height
    pts = pts.astype(np.int32)

    cv2.fillPoly(img, pts=[pts], color=color, lineType=cv2.LINE_4)


def draw_text(
    img: np.ndarray, text: str, p: np.ndarray, font, scale, color, thickness=None
):
    height, width, _ = img.shape
    dims = np.array([width, height])
    p = (p * dims).astype(np.int32)

    return cv2.putText(img, text, p, font, scale, color, thickness)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def save_pickle(path: str, data):
    with open(path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str):
    with open(path, "rb") as input_file:
        data = pickle.load(input_file)
    return data
