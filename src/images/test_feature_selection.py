from src.images.method_experiments import (
    test_rfe,
    test_sfs,
    test_anova,
)
import argparse
from src.images.method_experiments import prepare_features, estimators, estimators_anova
from src.images.features import (
    KarolewskiFilteredFeaturesExtractor,
    KarolewskiFeaturesExtractor,
    CustomFeaturesExtractor,
    EmptyFeaturesExtractor,
)
import src.images.utils as utils
from src.images.detector import get_face_detector
from src.images.preprocess import extract_faces_denta, extract_landmarks
from src.images.datasets import (
    get_denta_labels,
    get_dynamic_features,
)
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--selection",
        required=True,
        type=str,
        choices=["RFE", "ANOVA", "TOP", "SFS"],
    )
    parser.add_argument(
        "-rf",
        "--ref-points",
        required=False,
        type=str,
        choices=["0-32", "64-68"],
        default="0-32",
    )
    args = parser.parse_args()

    dynamic_data = get_dynamic_features("../data/denta_v1/ex3_data.csv")

    geom_features = [
        {
            "name": "Custom",
            "extra_features": dynamic_data,
            "cls": CustomFeaturesExtractor,
        },
        {
            "name": "Karolewski",
            "extra_features": dynamic_data,
            "cls": KarolewskiFeaturesExtractor,
        },
        {
            "name": "KarolewskiFiltered+Dynamic",
            "cls": KarolewskiFilteredFeaturesExtractor,
            "extra_features": dynamic_data,
            "add_extra_features": True,
        },
        {
            "name": "Dynamic",
            "cls": EmptyFeaturesExtractor,
            "extra_features": dynamic_data,
            "add_extra_features": True,
        },
    ]

    app = get_face_detector()
    print("Preparing DENTA dataset...")
    dataset_path = "../../data/denta_v1"
    df = get_denta_labels(dataset_path, split_type="random")
    extract_faces_denta(dataset_path, df, detector=lambda image: app.get(image))
    data_path = os.path.join("..", "..", "experiments", "images_results")
    extract_landmarks(dataset_path, df)

    features = []
    features = geom_features
    rfe_step = 0.1

    ref_points = tuple(map(int, args.ref_points.split("-")))
    print("Using reference points: ", ref_points)

    features = prepare_features(
        dataset_path, features, df=df, reference_points=ref_points
    )

    tests = []
    if args.selection == "ANOVA":
        print("Testing ANOVA")
        tests = test_anova(
            estimators_anova, features, range_to_test=range(1, 20), stratified=True
        )
    elif args.selection == "RFE":
        print("Testing RFE")
        tests = test_rfe(
            estimators, features, range_to_test=range(1, 20), step=rfe_step
        )
    elif args.selection == "SFS":
        print("Testing SFS")
        tests = test_sfs(
            estimators, features, features_num=19, iterations=20, inner_cv=0
        )
    else:
        print("ERROR")
        return

    ref_points_str = "_".join(map(str, ref_points))
    path = os.path.join(
        data_path,
        f"features_selection_{args.selection.lower()}_{ref_points_str}.pkl",
    )
    utils.save_pickle(path, tests)


main()
