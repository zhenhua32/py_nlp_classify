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


def main_pipeline_tfidf():
    """
    尝试下 tfidf 的最佳参数
    """
    train_df, dev_df, label2id = load_dataset()
    # 用 Pipeline 封装模型
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", RidgeClassifier())
    ])
    # 用 GridSearchCV 尝试最佳参数
    param_grid = {
        "tfidf__analyzer": ["word", "char"],
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3), (1, 4)],
        "tfidf__max_features": [10000, 20000, 30000],
        "tfidf__lowercase": [True, False],
        "tfidf__max_df": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "tfidf__norm": ["l1", "l2"],
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=8, verbose=2)
    grid_search.fit(train_df["text"], train_df["label_id"])
    print("best params: {}".format(grid_search.best_params_))
    print("best score: {:.4f}".format(grid_search.best_score_))
    # 保存最佳参数
    with open("./best_params.json", "w", encoding="utf-8") as f:
        json.dump(grid_search.best_params_, f, ensure_ascii=False, indent=4)

    # 用最佳参数重新训练模型
    best_pipeline = grid_search.best_estimator_
    best_pipeline.fit(train_df["text"], train_df["label_id"])
    pred_labels = best_pipeline.predict(dev_df["text"])
    print("在 dev 上的结果: ")
    print("f1 score: {:.4f}".format(f1_score(dev_df["label_id"], pred_labels, average="micro")))
    print("accuracy score: {:.4f}".format(accuracy_score(dev_df["label_id"], pred_labels)))


if __name__ == "__main__":
    # main_pipeline_tfidf()
    main_sample()
