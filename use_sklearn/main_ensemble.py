import os
import json
import time

import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier,
    HistGradientBoostingClassifier,
)
from tools import timeit, load_dataset


def get_features(train_df: pd.DataFrame, dev_df: pd.DataFrame, **kwargs):
    """
    获取特征
    kwargs 传入 TfidfVectorizer 的参数
    """
    # 通用 pipeline 跑出来的最佳参数
    kwargs["analyzer"] = kwargs.get("analyzer", "char")
    kwargs["ngram_range"] = kwargs.get("ngram_range", (1, 2))
    kwargs["max_features"] = kwargs.get("max_features", 30000)
    kwargs["lowercase"] = kwargs.get("lowercase", False)
    kwargs["max_df"] = kwargs.get("max_df", 0.5)
    kwargs["norm"] = kwargs.get("norm", "l2")

    tfidf = TfidfVectorizer(**kwargs)
    train_features = tfidf.fit_transform(train_df["text"])
    dev_features = tfidf.transform(dev_df["text"])

    return train_features, dev_features


@timeit
def run_ensemble_model(model, train_features, train_df, dev_features, dev_df):
    """
    运行模型
    """
    model.fit(train_features, train_df["label_id"])
    pred_labels = model.predict(dev_features)
    print("f1 score: {:.4f}".format(f1_score(dev_df["label_id"], pred_labels, average="micro")))
    print("accuracy score: {:.4f}".format(accuracy_score(dev_df["label_id"], pred_labels)))


def main_ensemble():
    """
    集成算法
    """
    train_df, dev_df, label2id = load_dataset()
    train_features, dev_features = get_features(train_df, dev_df)

    print("使用 AdaBoostClassifier")
    # TODO: 奇怪, 为什么集成算法更差了?
    model = AdaBoostClassifier(
        RidgeClassifier(),
        n_estimators=10,
        algorithm="SAMME",
    )
    run_ensemble_model(model, train_features, train_df, dev_features, dev_df)

    print("使用 BaggingClassifier")
    model = BaggingClassifier(
        RidgeClassifier(),
        n_estimators=10,
    )
    run_ensemble_model(model, train_features, train_df, dev_features, dev_df)

    print("使用 RandomForestClassifier")
    model = RandomForestClassifier(
        n_estimators=10,
    )
    run_ensemble_model(model, train_features, train_df, dev_features, dev_df)

    print("使用 ExtraTreesClassifier")
    model = ExtraTreesClassifier(
        n_estimators=10,
    )
    run_ensemble_model(model, train_features, train_df, dev_features, dev_df)

    print("使用 GradientBoostingClassifier")
    model = GradientBoostingClassifier(
        n_estimators=10,
    )
    run_ensemble_model(model, train_features, train_df, dev_features, dev_df)

    print("使用 HistGradientBoostingClassifier")
    model = HistGradientBoostingClassifier(
        max_iter=10,
    )
    run_ensemble_model(model, train_features, train_df, dev_features, dev_df)

    print("使用 StackingClassifier")
    model = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=10)),
            ("et", ExtraTreesClassifier(n_estimators=10)),
            ("gb", GradientBoostingClassifier(n_estimators=10)),
            ("hgb", HistGradientBoostingClassifier(max_iter=10)),
        ],
        final_estimator=RidgeClassifier(),
    )
    run_ensemble_model(model, train_features, train_df, dev_features, dev_df)

    print("使用 VotingClassifier")
    model = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=10)),
            ("et", ExtraTreesClassifier(n_estimators=10)),
            ("gb", GradientBoostingClassifier(n_estimators=10)),
            ("hgb", HistGradientBoostingClassifier(max_iter=10)),
        ],
        voting="soft",
    )
    run_ensemble_model(model, train_features, train_df, dev_features, dev_df)


if __name__ == "__main__":
    main_ensemble()
