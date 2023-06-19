from typing import Optional
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from src.data_modules import DentaDataModule, TNFDataModule
import numpy as np
from itertools import product
from sklearn.base import clone
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from src.evaluation import get_importances
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier


estimators = [
    {
        'name': 'XGB',
        'estimator': Pipeline([
            ('scaler', StandardScaler()), 
            # ('scaler', MinMaxScaler()),
            ('clf', XGBClassifier(
                learning_rate=0.3,
                # n_estimators=100,
                max_depth=6,
                min_child_weight=1,
                use_label_encoder=False,
                objective="binary:logistic",
                eval_metric='auc',
                subsample=1,
                random_state=np.random.RandomState(None)
            ))
        ]),
        'importance_getter': 'named_steps.clf.feature_importances_',
    },
    {
        'name': 'SVM',
        'estimator': Pipeline([
            ('scaler', StandardScaler()),
            # ('scaler', MinMaxScaler()),
            ('clf', LinearSVC(max_iter=2000, random_state=np.random.RandomState(None)))
        ]),
        'importance_getter': 'named_steps.clf.coef_',
    },
    {
        'name': 'RLR',
        'estimator': Pipeline([
            ('scaler', StandardScaler()),
            # ('scaler', MinMaxScaler()),
            ('clf', LogisticRegression(random_state=np.random.RandomState(
                None), #dual=True, solver='liblinear'
                ))
        ]),
        'importance_getter': 'named_steps.clf.coef_',
    },
]

dummy_estimators = [
    {
        'name': 'Uniform',
        'estimator': Pipeline([
            ('clf', DummyClassifier(strategy='uniform'))
        ]),
    },
    {
        'name': 'Stratified',
        'estimator': Pipeline([
            ('clf', DummyClassifier(strategy='stratified'))
        ]),
    },
    {
        'name': 'MostFrequent',
        'estimator': Pipeline([
            ('clf', DummyClassifier(strategy='most_frequent'))
        ]),
    }
]


def prepare_features(path: str, features: list[dict], df: Optional[pd.DataFrame] = None, reference_points: tuple[int, int] = (0, 32)) -> list[dict]:
    new_features = []
    for feature in features:
        dm = DentaDataModule(
            data_dir=path, features_extractor=feature['cls'](reference_points=reference_points), df=df)
        dm.prepare_data()
        # Keep only values present in extra features
        extra_features_data = []
        if 'extra_features' in feature:
            keys_to_keep = set(list(feature['extra_features'].keys()))
            keys = dm.data.features_grouped.keys()
            for key in list(keys):
                if key not in keys_to_keep:
                    del dm.data.features_grouped[key]
                    del dm.data.targets_grouped[key]
                elif feature.get('add_extra_features', False):
                    extra_features_data.append(feature['extra_features'][key])
        
        X, y, = dm.aggregate()
        new_feature = feature.copy()
        new_feature['X_avg'] = X
        new_feature['X_concat'] = dm.aggregate(0)[0]
        new_feature['X_neutral'] = dm.filter_photos(0)[0]
        new_feature['X_smile'] = dm.filter_photos(2)[0]
        new_feature['y'] = y
        new_feature['ref_points'] = '_'.join(map(str, reference_points))
        if len(extra_features_data) > 0:
            extra_features_data = np.array(extra_features_data)
            new_feature['X_avg'] = np.hstack([new_feature['X_avg'], extra_features_data])
            new_feature['X_concat'] = np.hstack([new_feature['X_concat'], extra_features_data])
            new_feature['X_neutral'] = np.hstack([new_feature['X_neutral'], extra_features_data])
            new_feature['X_smile'] = np.hstack([new_feature['X_smile'], extra_features_data])

        new_features.append(new_feature)

    return new_features


def count_feature_occurrences(df: pd.DataFrame, estimator: str, features: str, test_name: str) -> pd.DataFrame:
    feats = df[(df['estimator'] == estimator) & (df['features'] == features) & (
        df['test_name'] == test_name)].iloc[0]['features_idx_folds']
    unique, counts = np.unique(feats, return_counts=True)
    counts = dict(zip(unique, counts))
    count_values = np.array(list(counts.values()))
    count_values = count_values / count_values.max() * 100
    counts_df = pd.DataFrame({'feature': counts.keys(), 'count': count_values}).sort_values(
        by='count', ascending=False).reset_index(drop=True)
    return counts_df


def test_concat_vs_avg(estimators: list[dict], features: list[dict], cv=5, iterations: int = 20, stratified: bool = False):
    tests = []

    versions = ['avg', 'concat', 'neutral', 'smile']
    combinations = list(product(versions, estimators, features))
    with tqdm(total=len(combinations)*iterations) as t:
        for iter in range(iterations):
            kf = StratifiedKFold(
                cv, shuffle=True) if stratified else KFold(cv, shuffle=True)
            splits = list(kf.split(features[0]['X_avg'], features[0]['y']))
            for version, estimator, feature in combinations:
                test_sample = {
                    'estimator': estimator['name'],
                    'features': feature['name'],
                    'ref_points': feature['ref_points'],
                    'aggregation': version,
                    'iter': iter
                }

                X, y = feature[f'X_{version}'], feature['y']
                for train, test in splits:
                    test_sample_copy = test_sample.copy()
                    X_train, X_test = X[train], X[test]
                    y_train, y_test = y[train], y[test]
                    model = clone(estimator['estimator'])
                    model = model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    if 'selector' in model.named_steps:
                        features_idx = model.named_steps['selector'].get_feature_names_out(
                            range(feature[f'X_{version}'].shape[1])).tolist()
                        test_sample_copy['features_num'] = model.named_steps['selector'].k
                        test_sample_copy['features_idx'] = features_idx

                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_test, preds, average='macro', zero_division=0)

                    test_sample_copy['f1'] = f1
                    test_sample_copy['precision'] = precision
                    test_sample_copy['recall'] = recall
                    test_sample_copy['predictions'] = preds.tolist()
                    test_sample_copy['true'] = y_test.tolist()
                    test_sample_copy['split'] = (train, test)
                    tests.append(test_sample_copy)
                t.update(1)
    return tests