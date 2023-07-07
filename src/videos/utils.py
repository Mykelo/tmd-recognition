import cv2
import os, shutil
from os import path
import numpy as np
from tqdm.notebook import tqdm
from numpy import ones, vstack
from numpy.linalg import lstsq, norm
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from scipy import stats
import warnings
import nolds


def show_img(img: np.ndarray, color=False, axis=False, figsize=(10, 10)):
    if color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=figsize)
    if not axis:
        plt.axis("off")
    ax.imshow(img)


def get_raw_frames(video_path: str, frames_idx: list[int] = None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)

    frames = []
    idx = 0
    succ, frame = cap.read()
    while succ:
        if frames_idx is None or idx in frames_idx:
            frames.append(frame)
        idx += 1
        succ, frame = cap.read()

    return frames


def get_faces_coords(frames: np.ndarray, face_detector):
    faces_coords = []
    for frame in tqdm(
        frames,
        total=len(frames),
        position=1,
        leave=False,
        desc="Extracting faces coordinates from video frames",
    ):
        faces = face_detector(frame)
        x1, y1, x2, y2 = max(faces, key=lambda face: face["bbox"][3] - face["bbox"][1])[
            "bbox"
        ]
        faces_coords.append((x1, y1, x2, y2))

    return faces_coords


def find_max_size(coords):
    max_dims = []
    for c in coords:
        width = c[2] - c[0]
        height = c[3] - c[1]
        max_dim = max(width, height)
        max_dims.append(max_dim)

    return max(max_dims)


def extract_faces(
    frames: np.ndarray,
    coords: np.ndarray,
    target_size: int,
    save_path: str = None,
    fill_to: int = None,
    padding=80,
    silent=False,
) -> np.ndarray:
    if save_path:
        frames_dir = os.path.join(save_path, "frames")
        if path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir)

    faces = []
    for idx, (frame, c) in tqdm(
        enumerate(zip(frames, coords)),
        total=len(frames),
        position=1,
        leave=False,
        desc="Extracting faces from video frames",
        disable=silent,
    ):
        image_height, image_width, _ = frame.shape

        x1, y1, x2, y2 = c
        width = x2 - x1
        height = y2 - y1

        x1 -= padding
        x2 += padding

        y1 -= padding
        y2 += fill_to + padding - height

        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, image_width - 1)
        y2 = min(y2, image_height - 1)

        image_crop = frame[int(y1) : int(y2), int(x1) : int(x2), :]

        new_width = x2 - x1
        new_height = y2 - y1

        scaler = new_width / new_height
        face_img = cv2.resize(image_crop, (int(scaler * target_size), target_size))

        faces.append(face_img)

        if save_path is not None:
            file_path = os.path.join(frames_dir, f"{idx}.jpg")
            cv2.imwrite(file_path, face_img)

    return faces


def extract_face(img: np.ndarray, target_size: int, detector) -> np.array:
    image_height, image_width, _ = img.shape

    faces = detector(img)
    x1, y1, x2, y2 = max(faces, key=lambda face: face["bbox"][3] - face["bbox"][1])[
        "bbox"
    ]

    width = x2 - x1
    height = y2 - y1
    dim_diff = height - width
    width_fill = max(dim_diff, 0)
    height_fill = -min(dim_diff, 0)

    scale = 0.2
    x1 -= width_fill / 2 + width * scale / 2
    y1 -= height_fill / 2 + height * scale / 2
    x2 += width_fill / 2 + width * scale / 2
    y2 += height_fill / 2 + height * scale / 2

    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, image_width - 1)
    y2 = min(y2, image_height - 1)

    image_crop = img[int(y1) : int(y2), int(x1) : int(x2), :]
    face_img = cv2.resize(image_crop, (target_size, target_size))

    return face_img


def get_face_detector(det_size: tuple[int, int] = (256, 256)) -> FaceAnalysis:
    with HiddenPrints():
        app = FaceAnalysis(name="antelope", root="~/.insightface")
        app.prepare(ctx_id=0, det_size=det_size)

    return app


def extract_landmarks(
    images: np.ndarray, detector, save_path: str = None, silent=False
) -> np.ndarray:
    landmarks_ = []
    for img in tqdm(
        images,
        total=len(images),
        position=1,
        leave=False,
        desc="Extracting landmarks",
        disable=silent,
    ):
        landmarks = detector.get_landmarks(img)
        landmarks_.append(landmarks)

    landmarks_ = np.stack(landmarks_, axis=0)
    if save_path:
        file_path = os.path.join(save_path, "landmarks.npy")
        np.save(file_path, landmarks_)

    return landmarks_


def extract_landmarks_v1(img: np.ndarray, detector) -> np.array:  # detector type
    landmarks = detector.get_landmarks(img)
    return landmarks


def landmarks_to_points(
    images: np.ndarray, landmarks: np.ndarray, save_path: str = None, silent=False
) -> np.ndarray:
    points = []

    for img, landmarks_ in tqdm(
        zip(images, landmarks),
        total=len(images),
        position=1,
        leave=False,
        desc="Converting landmarks to points",
        disable=silent,
    ):
        frame_points = []
        for i in range(landmarks_.shape[0] // 2):
            x_pred = int(landmarks_[i * 2] * img.shape[1])
            y_pred = int(landmarks_[i * 2 + 1] * img.shape[0])
            frame_points.append((x_pred, y_pred))
        points.append(frame_points)

    points = np.array(points)
    if save_path is not None:
        file_path = os.path.join(save_path, "points.npy")
        np.save(file_path, points)

    return points


def landmarks_to_points_v1(img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    points = []
    for i in range(landmarks.shape[0] // 2):
        x_pred = int(landmarks[i * 2] * img.shape[1])
        y_pred = int(landmarks[i * 2 + 1] * img.shape[0])
        points.append((x_pred, y_pred))

    return np.array(points)


def preprocess_video(
    video_path: str,
    face_detector,
    landmarks_detector,
    target_size: int,
    padding=80,
    save_path: str = None,
):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    frames = get_raw_frames(video_path)
    faces_coords = get_faces_coords(frames, lambda img: face_detector.get(img))
    max_size = find_max_size(faces_coords)
    faces = extract_faces(
        frames, faces_coords, target_size, save_path, max_size, padding
    )
    landmarks = extract_landmarks(faces, landmarks_detector, save_path)
    points = landmarks_to_points(faces, landmarks, save_path)

    return faces, landmarks, points


def draw_vertical_line_through_point(img, point, line_color=(255, 0, 0)):
    temp_img = img.copy()
    cv2.line(temp_img, (point[0], img.shape[1]), (point[0], 0), line_color, 2)
    return temp_img


def get_lines(points: np.ndarray, line_v: str = "v4", line_h: str = "h1"):
    if line_h == "h1":
        hline = (points[64], points[68])
    elif line_h == "h2":
        hline = (points[76], points[82])
    elif line_h == "h3":
        p1 = get_avg_point([points[60], points[64]])
        p2 = get_avg_point([points[68], points[72]])
        hline = (p1, p2)
    else:
        raise Exception(f"line_h: '{line_h}' not available")

    vline = None
    if line_v == "v0":
        hline = None
        vline = (points[51], points[57])
    elif line_v == "v1":
        thr_point = points[51]
    elif line_v == "v2":
        thr_point = points[57]
    elif line_v == "v3":
        thr_point = points[79]
    elif line_v == "v4":
        thr_point = points[90]
    elif line_v == "v5":
        thr_point = get_avg_point([points[64], points[68]])
    elif line_v == "v6":
        thr_point = get_avg_point([points[51], points[57]])
    elif line_v == "v7":
        thr_point = get_avg_point([points[79], points[90]])
    elif line_v == "v8":
        thr_point = get_avg_point([points[79], points[90], points[51], points[57]])
    else:
        raise Exception(f"line_v: '{line_v}' not available")

    if vline is None:
        vline = get_perpendicular_line(hline, thr_point)

    return vline, hline


def draw_lines(
    img_: np.ndarray,
    points: np.ndarray,
    line_v: str = "v4",
    line_h: str = "h1",
    landmarks: bool = False,
    index: bool = True,
    thickness=2,
    line_color=(0, 0, 255),
):
    img = img_.copy()
    if landmarks:
        img = draw_landmarks(img, points, index)

    line_v, line_h = get_lines(points, line_v=line_v, line_h=line_h)
    if line_h is not None:
        img = draw_line_through_points(
            line_h[0], line_h[1], img, thickness, line_color=line_color
        )
    img = draw_line_through_points(
        line_v[0], line_v[1], img, thickness, line_color=line_color
    )

    return img


def draw_landmarks(
    img: np.ndarray,
    points: np.ndarray,
    index: bool = False,
    index_color=(0, 255, 0),
    point_color=(0, 255, 0),
    point_size=4,
    index_size=0.6,
) -> np.ndarray:
    temp_img = img.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(temp_img, (x, y), point_size, point_color, -1)
        if index:
            cv2.putText(
                temp_img,
                str(i),
                (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                index_size,
                index_color,
                1,
            )

    temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
    return temp_img


def draw_line_through_points(
    p1, p2, img: np.ndarray, line_color, thickness=2
) -> np.ndarray:
    temp_img = img.copy()
    points = [p1, p2]
    x_coords, y_coords = zip(*points)
    height, width, _ = temp_img.shape
    A = vstack([x_coords, ones(len(x_coords))]).T
    #  m not 0
    a, b = lstsq(A, y_coords, rcond=None)[0]
    y0 = int(b)
    ymax = int(a * width + b)

    cv2.line(temp_img, (0, y0), (width, ymax), line_color, thickness)
    return temp_img


def get_perpendicular_line(
    line, point
) -> tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int]]]:
    x_coords, y_coords = zip(*line)
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords, rcond=None)[0]
    #  m not 0
    mp = -1.0 / m
    cp = point[1] - (mp * point[0])
    px = round(-cp / mp)
    qx = round((point[1] - cp) / mp)

    return ((px, 0), (qx, point[1]))


def draw_perpendicular_line(
    line, point, img: np.ndarray
) -> tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int]]]:
    temp_img = img.copy()
    x_coords, y_coords = zip(*line)
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords, rcond=None)[0]
    mp = -1.0 / m
    cp = point[1] - (mp * point[0])
    px = round(-cp / mp)
    qx = round((len(temp_img) - cp) / mp)

    cv2.line(temp_img, (px, 0), (qx, len(temp_img)), (255, 0, 0), 2)

    return temp_img, ((px, 0), (qx, len(temp_img)))


def distance_from_line(
    line: tuple[tuple[int, int], tuple[int, int]], point: tuple[int, int]
) -> tuple[float, int]:
    p1, p2 = np.array(line)
    side = 1 if np.cross(p2 - p1, p1 - point) > 0 else -1  # check it

    return norm(np.cross(p2 - p1, p1 - point)) / norm(p2 - p1), side


def get_avg_point(points):
    sum_x = sum([p[0] for p in points])
    sum_y = sum([p[1] for p in points])
    avg_x = int(sum_x / len(points))
    avg_y = int(sum_y / len(points))
    return avg_x, avg_y


def extract_points_from_video(
    video_path: str,
    face_detector: FaceAnalysis,
    landmarks_detector,
    FACE_SIZE: int = 256,
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)  # 90 frames

    all_points = []
    succ, frame = cap.read()
    while succ:
        face_img = extract_face(frame, FACE_SIZE, lambda img: face_detector.get(img))
        landmarks = extract_landmarks(face_img, landmarks_detector)
        points = landmarks_to_points(face_img, landmarks)
        all_points.append(points)
        succ, frame = cap.read()

    return np.stack(all_points, axis=0)


def get_frames(
    video_path: str,
    frames_idx: list[int],
    face_detector,
    landmarks_detector,
    face_size,
    lines=None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    cap = cv2.VideoCapture(video_path)

    frames = []
    points_ = []
    idx = 0
    succ, frame = cap.read()
    while succ:
        if idx in frames_idx:
            face_img = extract_face(
                frame, face_size, lambda img: face_detector.get(img)
            )
            landmarks = extract_landmarks(face_img, landmarks_detector)
            points = landmarks_to_points(face_img, landmarks)
            if lines:
                landmark_face = draw_landmarks(face_img, landmarks, index=True)
                if lines == "v1":
                    face_img = draw_line_through_points(
                        points[51], points[57], landmark_face
                    )
                else:
                    line_h1_img = draw_line_through_points(
                        points[64], points[68], landmark_face
                    )
                    line = (points[64], points[68])
                    face_img, perp_line = draw_perpendicular_line(
                        line, points[51], line_h1_img
                    )
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            frames.append(face_img)
            points_.append(points)
        idx += 1
        succ, frame = cap.read()

    return frames, points_


def get_motion_patterns(
    video_points,
    line_v="v4",
    line_h="h1",
    central_point=16,
    lowpass_kernel=None,
    stabilize=False,
):
    motion_patterns = {
        "abduction": [],
        "abduction_deviation": [],
    }
    for idx, points in enumerate(video_points):
        vline, _ = get_lines(points, line_v=line_v, line_h=line_h)

        if isinstance(central_point, list):
            distances = [distance_from_line(vline, points[p]) for p in central_point]
            total_dist = np.mean(
                [dist_dev * side_dev for dist_dev, side_dev in distances]
            )
        else:
            dist_dev, side_dev = distance_from_line(vline, points[central_point])
            total_dist = dist_dev * side_dev

        dist_y = distance(points[90], points[94])
        motion_patterns["abduction_deviation"].append(total_dist)
        motion_patterns["abduction"].append(dist_y)

    if lowpass_kernel is not None:
        motion_patterns = {
            k: np.convolve(v, lowpass_kernel, mode="same")
            for k, v in motion_patterns.items()
        }

    if stabilize:
        stabilizer = motion_patterns["abduction_deviation"][0]

    return motion_patterns


def load_frames(data_path: str, idxs: list[int]) -> np.ndarray:
    frames_path = path.join(data_path, "frames")
    frames = []
    for idx in idxs:
        file_path = path.join(frames_path, f"{idx}.jpg")
        frame = cv2.imread(file_path)
        frames.append(frame)

    return np.array(frames)


def generate_report(
    data_path: str,
    motion_patterns,
    features,
    points,
    line_v: str,
    line_h: str,
    landmarks: bool,
    index: bool,
    figsize,
):

    abduction_deviation = motion_patterns["abduction_deviation"]
    abduction = motion_patterns["abduction"]
    normalized_abd = (abduction - min(abduction)) / (max(abduction) - min(abduction))

    max_right_dev_fr = np.argmax(abduction_deviation)
    max_left_dev_fr = np.argmin(abduction_deviation)
    pivot = np.argmin(normalized_abd)

    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=figsize)

    ax1.set_title("Normalized obduction pattern (chin from eyes)")
    ax1.set_xlabel("Frames")
    ax1.set_ylabel("Distance")
    ax1.plot(normalized_abd)
    ax1.axvline(x=pivot, color="g", linestyle="dashed")
    ax1.annotate("open  close", xy=(pivot, 1), xytext=(pivot - 5.5, 0.96))
    ax1.grid()

    ax2.set_title("Abduction deviation pattern")
    ax2.set_xlabel("Frames")
    ax2.set_ylabel("Deviation (in pixels)")
    ax2.plot(abduction_deviation)
    ax2.axhline(y=0, color="r", linestyle="dashed")
    ax2.annotate("open  close", xy=(pivot, 1), xytext=(pivot - 5.5, 3.5))
    ax2.axvline(x=pivot, color="g", linestyle="dashed")
    ax2.grid()

    frames_idx = [0, pivot, max_left_dev_fr, max_right_dev_fr]

    frames = load_frames(data_path, frames_idx)
    if lines is not None:
        frames = [
            draw_lines(frame, points[idx], line_v, line_h, landmarks, index)
            for frame, idx in zip(frames, frames_idx)
        ]

    ax3.imshow(frames[0])
    ax3.set_title("Closed mouth")

    ax4.imshow(frames[2])
    ax4.set_title("Max left deviation")

    ax5.imshow(frames[1])
    ax5.set_title("Opened mouth")

    ax6.imshow(frames[3])
    ax6.set_title("Max right deviation")

    print(f"Change motion in frame: {pivot}")
    print(
        f"Max right deviation (in pixels): {features['max_right_dev']} for frame {max_right_dev_fr}"
    )
    print(
        f"Max left deviation (in pixels): {features['max_left_dev']} for frame {max_left_dev_fr}"
    )
    print(f"Average right deviation (in pixels): {features['avg_right_dev']}")
    print(f"Average left deviation (in pixels): {features['avg_left_dev']}")

    print(f"Mean deviation (in pixels): {features['mean_dev']}")
    print(f"Median deviation (in pixels): {features['median_dev']}")
    print(f"Standard deviation (in pixels): {features['std_dev']}")
    print(f"Excess Kurtosis deviation: {features['excess_kurtosis_dev']}")
    print(f"Lyapunov exponent deviation: {features['lyapunov_exp_dev']}")
    print(
        f"Skewness coefficient of deviation (Pearson's second): {features['skewness_dev']}"
    )
    print()
    print(f"Mean abduction (normalized): {features['mean_abd']}")
    print(f"Median abduction (normalized): {features['median_abd']}")
    print(f"Standard deviation of abduction (normalized): {features['std_abd']}")
    print(f"Excess Kurtosis: {features['excess_kurtosis_abd']}")
    print(f"Lyapunov exponent: {features['lyapunov_exp_abd']}")
    print(f"Skewness coefficient (Pearson's second): {features['skewness_abd']}")


def distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def angle_3_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.arccos(cosine_angle)


def analyze_video(
    data_path: str,
    line_v="v1",
    line_h="h1",
    mp_kernel=None,
    show_report=True,
    landmarks=True,
    index=True,
    figsize=(20, 20),
):

    # Lyapunov exponent
    # https://nolds.readthedocs.io/en/latest/nolds.html?highlight=lyapunov#lyapunov-exponent-rosenstein-et-al

    video_points = np.load(os.path.join(data_path, "points.npy"))
    motion_patterns = get_motion_patterns(video_points, line_v=line_v, line_h=line_h)
    # add more options (not apply for all)
    if mp_kernel is not None:
        motion_patterns = {
            k: np.convolve(v, mp_kernel, mode="valid")
            for k, v in motion_patterns.items()
        }

    abduction_deviation = motion_patterns["abduction_deviation"]
    abduction = motion_patterns["abduction"]
    normalized_abd = (abduction - min(abduction)) / (max(abduction) - min(abduction))

    right_abd = list(filter(lambda x: x > 0, abduction_deviation))
    left_abd = list(filter(lambda x: x <= 0, abduction_deviation))

    mean_dev = np.mean(abduction_deviation)
    median_dev = np.median(abduction_deviation)
    std_dev = np.std(abduction_deviation)

    mean_abd = np.mean(normalized_abd)
    median_abd = np.median(normalized_abd)
    std_abd = np.std(normalized_abd)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lyapunov_exp_dev = nolds.lyap_r(abduction_deviation)
        lyapunov_exp_abd = nolds.lyap_r(normalized_abd)

    features = {
        "max_right_dev": max(abduction_deviation),
        "max_left_dev": abs(min(abduction_deviation)),
        "avg_right_dev": 0 if len(right_abd) == 0 else np.average(right_abd),
        "avg_left_dev": 0 if len(left_abd) == 0 else abs(np.average(left_abd)),
        "mean_dev": mean_dev,
        "median_dev": median_dev,
        "std_dev": std_dev,
        # <0 -> less outliers, 0 -> normal, >3 -> more outliers
        "excess_kurtosis_dev": stats.kurtosis(
            abduction_deviation
        ),  # degree of presence of outliers (fisher is already kurtosis-3 to make 0 for normal)
        "lyapunov_exp_dev": lyapunov_exp_dev,  # indicate chaos and unpredictability (positive exponent is a strong indicator for chaos)
        "skewness_dev": (3 * (mean_dev - median_dev))
        / std_dev,  # pearon's second coefficient of skewness ([-0.5, 0.5] -> nearly symmetrical, [-1, -0.5] or [0.5, 1] -> slightly skewed, <-1 or >1 -> extremely skewed)
        "mean_abd": mean_abd,
        "median_abd": median_abd,
        "std_abd": std_abd,
        "excess_kurtosis_abd": stats.kurtosis(normalized_abd),
        "lyapunov_exp_abd": lyapunov_exp_abd,
        "skewness_abd": (3 * (mean_abd - median_abd)) / std_abd,
    }

    if show_report:
        generate_report(
            data_path,
            motion_patterns,
            features,
            video_points,
            lines,
            landmarks,
            index,
            figsize,
        )

    return features
