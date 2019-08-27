"""
classification.preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module provides general preprocessing.
"""

from pathlib import Path
import random
import json
from typing import List, Tuple, Dict

import pandas as pd
from spacy.tokenizer import Tokenizer
from spacy.lang.de import German
from sklearn import model_selection


TOKENIZER = Tokenizer(German().vocab)
CURRENT_DIR = Path(__file__).parent.absolute()


def tokenize(document: str) -> List[str]:
    return [token.text for token in TOKENIZER(document)]


def load(
    corpus: str, split: bool = False, downsample: bool = True
) -> Dict[str, pd.DataFrame]:
    if corpus in {"dramen", "drama", "dramas"}:
        if downsample:
            if split:
                filepath = Path(
                    CURRENT_DIR,
                    "data",
                    "downsampled",
                    "split",
                    "dramen-downsampled-{}.json",
                )
            else:
                filepath = Path(
                    CURRENT_DIR,
                    "data",
                    "downsampled",
                    "sentences",
                    "dramen-downsampled.json",
                )
        else:
            if split:
                filepath = Path(
                    CURRENT_DIR,
                    "data",
                    "sentences",
                    "split",
                    "dramen-sentences-{}.json",
                )
            else:
                filepath = Path(
                    CURRENT_DIR, "data", "sentences", "full", "dramen-sentences.json"
                )
    elif corpus in {"romane", "roman", "novels", "novel"}:
        if downsample:
            if split:
                filepath = Path(
                    CURRENT_DIR,
                    "data",
                    "downsampled",
                    "split",
                    "romane-downsampled-{}.json",
                )
            else:
                filepath = Path(
                    CURRENT_DIR,
                    "data",
                    "downsampled",
                    "sentences",
                    "romane-downsampled.json",
                )
        else:
            if split:
                filepath = Path(
                    CURRENT_DIR, "data", "sentences", "split", "romane-sentences-{}json"
                )
            else:
                filepath = Path(
                    CURRENT_DIR, "data", "sentences", "full", "romane-sentences.json"
                )
    elif corpus in {"wikipedia"}:
        if downsample:
            if split:
                filepath = Path(
                    CURRENT_DIR,
                    "data",
                    "downsampled",
                    "split",
                    "wikipedia-downsampled-{}.json",
                )
            else:
                filepath = Path(
                    CURRENT_DIR,
                    "data",
                    "downsampled",
                    "sentences",
                    "wikipedia-downsampled.json",
                )
        else:
            if split:
                filepath = Path(
                    CURRENT_DIR,
                    "data",
                    "sentences",
                    "split",
                    "wikipedia-sentences-{}.json",
                )
            else:
                filepath = Path(
                    CURRENT_DIR, "data", "sentences", "full", "wikipedia-sentences.json"
                )
    elif corpus in {"zeitung", "zeitungsartikel", "newspaper"}:
        if downsample:
            if split:
                filepath = Path(
                    CURRENT_DIR,
                    "data",
                    "downsampled",
                    "split",
                    "zeitung-downsampled-{}.json",
                )
            else:
                filepath = Path(
                    CURRENT_DIR,
                    "data",
                    "downsampled",
                    "sentences",
                    "zeitung-downsampled.json",
                )
        else:
            if split:
                filepath = Path(
                    CURRENT_DIR,
                    "data",
                    "sentences",
                    "split",
                    "zeitung-sentences-{}.json",
                )
            else:
                filepath = Path(
                    CURRENT_DIR, "data", "sentences", "full", "zeitung-sentences.json"
                )
    else:
        raise ValueError(f"Corpus '{corpus}' does not exist.")
    if split:
        return {
            "train": pd.read_json(str(filepath).format("train")),
            "val": pd.read_json(str(filepath).format("val")),
            "test": pd.read_json(str(filepath).format("test")),
        }
    else:
        return pd.read_json(filepath)


def split(
    X: pd.Series,
    y: pd.Series,
    val: float = 0.1,
    test: float = 0.1,
    random_state: int = 23,
):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=test, random_state=random_state
    )
    X_train, X_val, y_train, y_val = model_selection.train_test_split(
        X_train, y_train, test_size=val, random_state=random_state
    )
    return (
        {"train": X_train, "val": X_val, "test": X_test},
        {"train": y_train, "val": y_val, "test": y_test},
    )


def split_and_export(directory, downsample_corpus=True):
    for file in Path(directory).glob("*.json"):
        corpus = pd.read_json(file)
        if downsample_corpus:
            corpus = downsample(corpus)
        X, y = split(corpus["text"], corpus["class"])
        for s in {"train", "test", "val"}:
            with open(f"{file.stem}-{s}.json", "w", encoding="utf-8") as f:
                dump = [{"text": t, "class": l} for t, l in zip(X[s], y[s])]
                f.write(json.dumps(dump, indent=2, ensure_ascii=False))


def convert_flair_data(directory):
    for file in Path(directory).glob("*.json"):
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        with Path(f"{file.stem}-flair.txt").open("w", encoding="utf-8") as f:
            nl = "\n"
            data = [
                f"__label__{instance['class']} {instance['text'].replace(nl, ' ')}"
                for instance in data
            ]
            f.write("\n".join(data))


def downsample(dataset, ratio={0: 296, 1: 304, 2: 181}):
    a = dataset[dataset["class"] == 0].reset_index()
    b = dataset[dataset["class"] == 1].reset_index()
    c = dataset[dataset["class"] == 2].reset_index()

    a_ = random.sample(range(a.shape[0]), ratio[0])
    b_ = random.sample(range(b.shape[0]), ratio[1])
    c_ = random.sample(range(c.shape[0]), ratio[2])

    a = a.iloc[a.index.isin(a_)]
    b = b.iloc[b.index.isin(b_)]
    c = c.iloc[c.index.isin(c_)]
    return a.append(b).append(c)
