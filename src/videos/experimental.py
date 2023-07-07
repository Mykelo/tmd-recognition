from scipy import stats
import warnings
import nolds
import os
import numpy as np

from utils import get_motion_patterns, distance


def analyze_video_exp(
    data_path: str,
    lines="v1",
    lowpass_kernel=[0.5, 0.5, 0.5],
    highpass_kernel=[-1, 1],
    mp_init_kernel=None,
    show_report=False,
    landmarks=True,
    index=True,
    figsize=(20, 20),
):
    video_points = np.load(os.path.join(data_path, "points.npy"))
    motion_patterns = get_motion_patterns(video_points, lines)
    if mp_init_kernel is not None:
        motion_patterns = {
            k: np.convolve(v, mp_init_kernel, mode="valid")
            for k, v in motion_patterns.items()
        }

    abduction_deviation = motion_patterns["abduction_deviation"]
    abduction = motion_patterns["abduction"]

    mean_abd = np.mean(abduction)
    std_abd = np.std(abduction)
    median_abd = np.median(abduction)
    norm_abd = (abduction - mean_abd) / std_abd
    mean_norm_abd = np.mean(norm_abd)
    std_norm_abd = np.std(norm_abd)

    high_norm_abd = np.convolve(norm_abd, highpass_kernel, mode="valid")
    mean_high_norm_abd = np.mean(high_norm_abd)
    std_high_norm_abd = np.std(high_norm_abd)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        f1_abd = nolds.lyap_r(norm_abd)

    a = 3
    min_interval_abd = mean_high_norm_abd - (a * std_high_norm_abd)
    max_interval_abd = mean_high_norm_abd + (a * std_high_norm_abd)
    outliers_abd = [
        idx
        for idx in range(len(high_norm_abd))
        if (
            high_norm_abd[idx] > max_interval_abd
            or high_norm_abd[idx] < min_interval_abd
        )
        and idx > 0
    ]
    f2_abd = len(outliers_abd)

    f3_abd = max(abs(high_norm_abd)) / std_high_norm_abd

    f4_abd = stats.kurtosis(abduction)

    f5_abd = stats.skew(abduction)

    peaks_vals_abd = [
        high_norm_abd[idx] - high_norm_abd[idx - 1] for idx in outliers_abd
    ]
    peaks_vals_sum_abd = sum([-1 if x < 0 else 1 for x in peaks_vals_abd])
    f6_abd = 0
    if peaks_vals_sum_abd != 0:
        f6_abd = -1 if peaks_vals_sum_abd < 0 else 1

    features = {
        "f1": f1_abd,  # lyapunov (abnormalities, chaotic motion)
        "f2": f2_abd,  # no. of outliers
        "f3": f3_abd,  # ratio beteen peak amplitude and variance of a signal (high - unexpected data)
        "f4": f4_abd,  # excess kurtosis (sharpness -> flatter than normal - <0, peaked than normal - >0)
        "f5": f5_abd,  # skewness (measure of asymmetry of a distribution)
        "f6": f6_abd,  # where are the most of peaks (1-open, 0-nopeak, -1-close)
    }

    return features


def analyze_video_exp_dev(
    data_path: str,
    lines="v1",
    lowpass_kernel=[0.5, 0.5, 0.5],
    highpass_kernel=[-1, 1],
    mp_init_kernel=None,
    show_report=False,
    landmarks=True,
    index=True,
    figsize=(20, 20),
):
    video_points = np.load(os.path.join(data_path, "points.npy"))
    motion_patterns = get_motion_patterns(video_points, lines)

    abduction_deviation = motion_patterns["abduction_deviation"]

    right_dev = np.array(list(filter(lambda x: x >= 0, abduction_deviation)))
    left_dev = np.array(list(filter(lambda x: x <= 0, abduction_deviation)))

    mean_right_dev = 0 if len(right_dev) == 0 else np.mean(right_dev)
    mean_left_dev = 0 if len(left_dev) == 0 else abs(np.mean(left_dev))

    std_right_dev = 0 if len(right_dev) == 0 else np.std(right_dev)
    std_left_dev = 0 if len(left_dev) == 0 else np.std(left_dev)

    mean_dev = np.mean(abduction_deviation)
    std_dev = np.std(abduction_deviation)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lyapunov_exp_dev = nolds.lyap_r(abduction_deviation)

    norm_dev_left = (
        0 if std_left_dev == 0 else (left_dev - mean_left_dev) / std_left_dev
    )
    norm_dev_right = (
        0 if std_right_dev == 0 else (right_dev - mean_right_dev) / std_right_dev
    )

    high_norm_dev_left = (
        [0]
        if len(left_dev) == 0
        else np.convolve(norm_dev_left, highpass_kernel, mode="valid")
    )
    mean_high_norm_dev_left = np.mean(high_norm_dev_left)
    std_high_norm_dev_left = np.std(high_norm_dev_left)

    high_norm_dev_right = (
        [0]
        if len(right_dev) == 0
        else np.convolve(norm_dev_right, highpass_kernel, mode="valid")
    )
    mean_high_norm_dev_right = np.mean(high_norm_dev_right)
    std_high_norm_dev_right = np.std(high_norm_dev_right)

    a = 3
    min_interval_dev_left = mean_high_norm_dev_left - (a * std_high_norm_dev_left)
    max_interval_dev_left = mean_high_norm_dev_left + (a * std_high_norm_dev_left)
    outliers_dev_left = [
        idx
        for idx in range(len(high_norm_dev_left))
        if (
            high_norm_dev_left[idx] > max_interval_dev_left
            or high_norm_dev_left[idx] < min_interval_dev_left
        )
        and idx > 0
    ]
    no_out_left = len(outliers_dev_left)

    min_interval_dev_right = mean_high_norm_dev_right - (a * std_high_norm_dev_right)
    max_interval_dev_right = mean_high_norm_dev_right + (a * std_high_norm_dev_right)
    outliers_dev_right = [
        idx
        for idx in range(len(high_norm_dev_right))
        if (
            high_norm_dev_right[idx] > max_interval_dev_right
            or high_norm_dev_right[idx] < min_interval_dev_right
        )
        and idx > 0
    ]
    no_out_right = len(outliers_dev_right)

    wsp_peak_left = (
        0
        if std_high_norm_dev_left == 0
        else (max(abs(high_norm_dev_left)) / std_high_norm_dev_left)
    )
    wsp_peak_right = (
        0
        if std_high_norm_dev_right == 0
        else (max(abs(high_norm_dev_right)) / std_high_norm_dev_right)
    )

    features = {
        "max_right_dev": max(abduction_deviation),
        "max_left_dev": abs(min(abduction_deviation)),
        "mean_right_dev": mean_right_dev,
        "mean_left_dev": mean_left_dev,
        "mean_dev": mean_dev,
        "std_dev": std_dev,
        "excess_kurtosis_dev": stats.kurtosis(abduction_deviation),
        "lyapunov_exp_dev": lyapunov_exp_dev,
        "skewness_dev": stats.skew(abduction_deviation),
        "wsp_peak_left": wsp_peak_left,
        "wsp_peak_right": wsp_peak_right,
        "no_out_left": no_out_left,
        "no_out_right": no_out_right,
    }

    return features


def get_frontal_in_out_features(abd, abd_dev, highpass_kernel):
    abs_abd_dev = np.abs(abd_dev)
    left_dev = np.array([x if x < 0 else 0 for x in abd_dev])
    right_dev = np.array([x if x > 0 else 0 for x in abd_dev])
    i = np.argmax(right_dev)
    j = np.argmin(left_dev)

    if i == 0:
        i += 1
    elif i == len(right_dev) - 1:
        i -= 1

    if j == 0:
        j += 1
    elif j == len(left_dev) - 1:
        j -= 1

    mean = np.mean(abd_dev)

    std = np.std(abd_dev)

    dev_diff = np.abs(abd_dev[0] - abd_dev[-1])

    max_left_dev = np.abs((left_dev[j - 1] + left_dev[j] + left_dev[j + 1]) / 3)
    max_right_dev = np.abs((right_dev[i - 1] + right_dev[i] + right_dev[i + 1]) / 3)
    max_dev = max(max_left_dev, max_right_dev)

    max_dev_idx = np.argmax(abs_abd_dev)
    if max_dev_idx <= len(abd) // 3:
        max_dev_phase = 0
    elif max_dev_idx > 2 * (len(abd) // 3):
        max_dev_phase = 2
    else:
        max_dev_phase = 1

    max_dev_ratio = min(max_left_dev, max_right_dev) / max_dev

    dev_area = sum(
        [np.abs(abd_dev[i] + abd_dev[i + 1]) / 2 for i in range(len(abd_dev) - 1)]
    )
    dev_area_left = sum(
        [np.abs(left_dev[i] + left_dev[i + 1]) / 2 for i in range(len(left_dev) - 1)]
    )
    dev_area_right = sum(
        [np.abs(right_dev[i] + right_dev[i + 1]) / 2 for i in range(len(right_dev) - 1)]
    )

    dev_area_ratio = min(dev_area_left, dev_area_right) / max(
        dev_area_left, dev_area_right
    )

    skewness = stats.skew(abd_dev)

    kurtosis = stats.kurtosis(abd_dev)

    a = 3
    mean_abd = np.mean(abd)
    std_abd = np.std(abd)
    norm_abd = (abd - mean_abd) / std_abd
    high_norm_abd = np.convolve(norm_abd, highpass_kernel, mode="valid")
    mean_high_norm_abd = np.mean(high_norm_abd)
    std_high_norm_abd = np.std(high_norm_abd)
    min_int_abd = mean_high_norm_abd - (a * std_high_norm_abd)
    max_int_abd = mean_high_norm_abd + (a * std_high_norm_abd)
    high_norm_abd = np.convolve(norm_abd, highpass_kernel, mode="valid")
    outliers = len(
        [
            idx
            for idx in range(len(high_norm_abd))
            if (high_norm_abd[idx] > max_int_abd or high_norm_abd[idx] < min_int_abd)
            and idx > 0
        ]
    )

    outliers_ratio = max(high_norm_abd) / std_high_norm_abd

    return {
        "dev_diff": dev_diff,
        "max_dev": max_dev,
        "max_dev_phase": max_dev_phase,
        "max_dev_ratio": max_dev_ratio,
        "dev_area": dev_area,
        "dev_area_ratio": dev_area_ratio,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "outliers": outliers,
        "outliers_ratio": outliers_ratio,
    }


def analyze_frontal_video(
    data_path: str,
    line_v="v1",
    line_h="h1",
    lowpass_kernel=None,
    highpass_kernel=[-1, 1],
    show_report=False,
    landmarks=True,
    index=True,
    figsize=(20, 20),
    central_point=16,
):
    video_points = np.load(os.path.join(data_path, "points.npy"))
    motion_patterns = get_motion_patterns(
        video_points,
        line_v=line_v,
        line_h=line_h,
        central_point=central_point,
        lowpass_kernel=lowpass_kernel,
    )

    abd = motion_patterns["abduction"]
    abd_dev = motion_patterns["abduction_deviation"]
    pivot = np.argmax(abd)

    start_idx = np.where(abd[:pivot] == np.min(abd[:pivot]))[0][-1]
    end_idx = pivot + np.where(abd[pivot:] == np.min(abd[pivot:]))[0][0]

    abd_out = abd[start_idx:pivot]
    abd_in = abd[pivot:end_idx]

    abd_dev_out = abd_dev[start_idx:pivot]
    abd_dev_in = abd_dev[pivot:end_idx]

    out_features = get_frontal_in_out_features(abd_out, abd_dev_out, highpass_kernel)
    in_features = get_frontal_in_out_features(abd_in, abd_dev_in, highpass_kernel)

    op_time = pivot - start_idx
    cl_time = end_idx - pivot
    time_ratio = min(op_time, cl_time) / max(op_time, cl_time)

    p1 = video_points[pivot][90]
    p2 = video_points[pivot][94]
    motion_range = distance(p1, p2)

    features = {"time_ratio": time_ratio, "motion_range": motion_range}
    out_features = {f"open_{k}": v for k, v in out_features.items()}
    in_features = {f"close_{k}": v for k, v in in_features.items()}

    features = features | out_features | in_features

    return features


def analyze_frontal_move_video(
    data_path: str,
    lines="v1",
    lowpass_kernel=None,
    highpass_kernel=[-1, 1],
    show_report=False,
    landmarks=True,
    index=True,
    figsize=(20, 20),
    central_point=16,
):
    video_points = np.load(os.path.join(data_path, "points.npy"))
    motion_patterns = get_motion_patterns(
        video_points, lines=lines, central_point=16, lowpass_kernel=lowpass_kernel
    )

    abd = motion_patterns["abduction"]
    abd_dev = motion_patterns["abduction_deviation"]

    front_pivot = np.argmax(abd)

    left_dev = np.array([x if x < 0 else 0 for x in abd_dev])
    right_dev = np.array([x if x > 0 else 0 for x in abd_dev])
    i = np.argmax(right_dev)
    j = np.argmin(left_dev)

    if i == 0:
        i += 1
    elif i == len(right_dev) - 1:
        i -= 1

    if j == 0:
        j += 1
    elif j == len(left_dev) - 1:
        j -= 1

    max_left_dev = np.abs((left_dev[j - 1] + left_dev[j] + left_dev[j + 1]) / 3)
    max_right_dev = np.abs((right_dev[i - 1] + right_dev[i] + right_dev[i + 1]) / 3)
    max_dev = max(max_left_dev, max_right_dev)

    max_dev_ratio = min(max_left_dev, max_right_dev) / max_dev

    dev_area = sum(
        [np.abs(abd_dev[i] + abd_dev[i + 1]) / 2 for i in range(len(abd_dev) - 1)]
    )
    dev_area_left = sum(
        [np.abs(left_dev[i] + left_dev[i + 1]) / 2 for i in range(len(left_dev) - 1)]
    )
    dev_area_right = sum(
        [np.abs(right_dev[i] + right_dev[i + 1]) / 2 for i in range(len(right_dev) - 1)]
    )

    dev_area_ratio = min(dev_area_left, dev_area_right) / max(
        dev_area_left, dev_area_right
    )

    op_time = front_pivot
    cl_time = len(abd) - front_pivot
    time_ratio = min(op_time, cl_time) / max(op_time, cl_time)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

    skewness = stats.skew(abd_dev)

    kurtosis = stats.kurtosis(abd_dev)

    a = 3
    mean_abd = np.mean(abd)
    std_abd = np.std(abd)
    norm_abd = (abd - mean_abd) / std_abd
    high_norm_abd = np.convolve(norm_abd, highpass_kernel, mode="valid")
    mean_high_norm_abd = np.mean(high_norm_abd)
    std_high_norm_abd = np.std(high_norm_abd)
    min_int_abd = mean_high_norm_abd - (a * std_high_norm_abd)
    max_int_abd = mean_high_norm_abd + (a * std_high_norm_abd)
    high_norm_abd = np.convolve(norm_abd, highpass_kernel, mode="valid")
    outliers = len(
        [
            idx
            for idx in range(len(high_norm_abd))
            if (high_norm_abd[idx] > max_int_abd or high_norm_abd[idx] < min_int_abd)
            and idx > 0
        ]
    )

    outliers_ratio = max(high_norm_abd) / std_high_norm_abd

    p1 = video_points[front_pivot][90]
    p2 = video_points[front_pivot][94]
    motion_range = distance(p1, p2)

    features = {
        "max_dev": max_dev,
        "max_dev_ratio": max_dev_ratio,
        "dev_area": dev_area,
        "dev_area_ratio": dev_area_ratio,
        "time_ratio": time_ratio,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "outliers": outliers,
        "outliers_ratio": outliers_ratio,
        "motion_range": motion_range,
    }

    return features
