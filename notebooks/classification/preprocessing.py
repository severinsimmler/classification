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
        if split:
            filepath = Path(CURRENT_DIR, "data", "split", "dramen-{}.json")
        else:
            filepath = Path(CURRENT_DIR, "data", "full", "dramen.json")
    elif corpus in {"romane", "roman", "novels", "novel"}:
        if downsample:
            filepath = Path(
                CURRENT_DIR, "data", "split-downsampled", "romane-downsampled-{}.json"
            )
        else:
            filepath = Path(CURRENT_DIR, "data", "split", "romane-{}json")
        if not split:
            filepath = Path(CURRENT_DIR, "data", "full", "romane.json")
    elif corpus in {"wikipedia"}:
        if downsample:
            filepath = Path(
                CURRENT_DIR,
                "data",
                "split-downsampled",
                "wikipedia-downsampled-{}.json",
            )
        else:
            filepath = Path(CURRENT_DIR, "data", "split", "wikipedia-{}.json")
        if not split:
            filepath = Path(CURRENT_DIR, "data", "full", "wikipedia.json")
    elif corpus in {"zeitung", "zeitungsartikel", "newspaper"}:
        if downsample:
            filepath = Path(
                CURRENT_DIR, "data", "split-downsampled", "zeitung-downsampled-{}.json"
            )
        else:
            filepath = Path(CURRENT_DIR, "data", "split", "zeitung-{}.json")
        if not split:
            filepath = Path(CURRENT_DIR, "data", "full", "zeitung.json")
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


def downsample(
    dataset: pd.DataFrame, ratio: Tuple[int] = (73, 57, 43), random_state: int = 23
):
    a, b, c = tuple(set(dataset["class"]))
    a = dataset[dataset["class"] == a]
    b = dataset[dataset["class"] == b]
    c = dataset[dataset["class"] == c]
    random.seed(random_state)
    a_ = random.sample(list(a.index), ratio[0])
    b_ = random.sample(list(b.index), ratio[1])
    c_ = random.sample(list(c.index), ratio[2])
    a = a.iloc[[True if _ in a_ else False for _ in a.index]]
    b = b.iloc[[True if _ in b_ else False for _ in b.index]]
    c = c.iloc[[True if _ in c_ else False for _ in c.index]]
    return a.append(b).append(c)

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


def split_and_export(directory, downsample_corpus=False):
    for file in Path(directory).glob("*.json"):
        corpus = pd.read_json(file)
        if downsample_corpus:
            corpus = downsample(corpus)
        X, y = split(corpus["text"], corpus["class"])
        for s in {"train", "test", "val"}:
            if downsample_corpus:
                filename = f"{file.stem}-downsampled-{s}.json"
            else:
                filename = f"{file.stem}-{s}.json"
            with open(filename, "w", encoding="utf-8") as f:
                dump = [{"text": t, "class": l} for t, l in zip(X[s], y[s])]
                f.write(json.dumps(dump, indent=2, ensure_ascii=False))

                
def convert_flair_data(filepath):
    with filepath.open("r", encoding="utf-8") as file:
        data = json.load(file)
    with open(f"{filepath.stem}-flair.txt", "w", encoding="utf-8") as file:
        nl = "\n"
        data = [f"__label__{instance['class']} {instance['text'].replace(nl, ' ')}" for instance in data]
        file.write("\n".join(data))
