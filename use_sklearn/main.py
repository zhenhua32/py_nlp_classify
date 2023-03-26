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
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier

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
def run_linear_model(train_features, train_labels, dev_features, dev_labels):
    """
    运行线性模型
    """
    model = RidgeClassifier()
    model.fit(train_features, train_labels)
    pred_labels = model.predict(dev_features)
    print("f1 score: {:.4f}".format(f1_score(dev_labels, pred_labels, average="micro")))
    print("accuracy score: {:.4f}".format(accuracy_score(dev_labels, pred_labels)))


def main_sample():
    """
    调用分类模型的示例
    """
    train_df, dev_df, label2id = load_dataset()
    train_features, dev_features = get_features(train_df, dev_df)
    run_linear_model(train_features, train_df["label_id"], dev_features, dev_df["label_id"])





def main_ensemble():
    """
    集成算法
    """
    train_df, dev_df, label2id = load_dataset()
    train_features, dev_features = get_features(train_df, dev_df)
    # TODO: 奇怪, 为什么集成算法更差了?
    model = AdaBoostClassifier(
        RidgeClassifier(),
        n_estimators=10,
        algorithm="SAMME",
    )
    model = BaggingClassifier(
        RidgeClassifier(),
        n_estimators=10,
    )
    model.fit(train_features, train_df["label_id"])
    pred_labels = model.predict(dev_features)
    print("f1 score: {:.4f}".format(f1_score(dev_df["label_id"], pred_labels, average="micro")))
    print("accuracy score: {:.4f}".format(accuracy_score(dev_df["label_id"], pred_labels)))


if __name__ == "__main__":
    # main_pipeline_tfidf()
    main_sample()
    main_ensemble()
