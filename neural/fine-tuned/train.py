from pathlib import Path
import collections
import operator
import logging
import json
import random
import sys

import numpy as np
import spacy
from spacy_pytorch_transformers import (
    PyTT_Language,
    PyTT_WordPiecer,
    PyTT_TokenVectorEncoder,
)
import torch

sys.path.insert(0, str(Path(".").absolute()))
from classification import preprocessing


is_using_gpu = spacy.prefer_gpu()
if is_using_gpu:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

spacy.util.fix_random_seed(23)

MAX_EPOCHS = 10
LEARN_RATE = 2e-5
MAX_BATCH_SIZE = 64
LABELS = ["0", "1", "2"]
DROPOUT = spacy.util.decaying(0.6, 0.2, 1e-4)


def cyclic_triangular_rate(min_lr, max_lr, period):
    it = 1
    while True:
        cycle = np.floor(1 + it / (2 * period))
        x = np.abs(it / period - 2 * cycle + 1)
        relative = max(0, 1 - x)
        yield min_lr + (max_lr - min_lr) * relative
        it += 1


def get_batches(train_data):
    """This will set the batch size to start at 1, and increase
    each batch until it reaches a maximum size.
    """
    MAX_BATCH_SIZE = 64
    if len(train_data) < 1000:
        MAX_BATCH_SIZE /= 2
    if len(train_data) < 500:
        MAX_BATCH_SIZE /= 2
    batch_size = spacy.util.compounding(1, MAX_BATCH_SIZE, 1.001)
    batches = spacy.util.minibatch(train_data, size=batch_size)
    return batches



if __name__ == "__main__":
    for corpus in ["dramen", "romane", "zeitung", "wikipedia"]:
        dataset = preprocessing.load(corpus, split=True, downsample=True)
        nlp = spacy.load("de_pytt_bertbasecased_lg")
        textcat = nlp.create_pipe(
            "pytt_textcat", config={"architecture": "softmax_last_hidden"}
        )
        for label in LABELS:
            textcat.add_label(label)

        X = dataset["train"]["text"]
        y = dataset["train"]["class"]

        train_data = list(
            zip(
                X,
                [
                    {
                        "cats": {
                            "0": 1 if label == 0 else 0,
                            "1": 1 if label == 1 else 0,
                            "2": 1 if label == 2 else 0,
                        }
                    }
                    for label in y
                ],
            )
        )

        nlp.add_pipe(textcat, last=True)

        optimizer = nlp.resume_training()
        optimizer.alpha = 0.001
        optimizer.pytt_weight_decay = 0.005
        optimizer.L2 = 0.0
        learn_rates = cyclic_triangular_rate(
            LEARN_RATE / 3, LEARN_RATE * 3, 2 * len(train_data) // MAX_BATCH_SIZE
        )
        losses = collections.Counter()

        SCORES = {"TEST": list(), "DEV": list()}
        try:
            for epoch in range(MAX_EPOCHS):
                logging.error(f"Epoch #{epoch + 1}")
                random.shuffle(train_data)
                batches = get_batches(train_data)
                for batch in batches:
                    optimizer.pytt_lr = next(learn_rates)
                    texts, annotations = zip(*batch)
                    dropout = next(DROPOUT)
                    nlp.update(
                        texts, annotations, sgd=optimizer, drop=dropout, losses=losses
                    )

                stats = list()
                for _, row in dataset["test"].iterrows():
                    t = nlp(row["text"])
                    pred = int(max(t.cats.items(), key=operator.itemgetter(1))[0])
                    if pred == row["class"]:
                        stats.append(1)
                    else:
                        stats.append(0)

                SCORES["TEST"].append(sum(stats) / len(dataset["test"]))
                print(max(SCORES["TEST"]))
            print(SCORES)
            nlp.to_disk(f"{corpus}-bert")
            print("TESTSET")
            for _, row in dataset["test"].iterrows():
                t = nlp(row["text"])
                pred = int(max(t.cats.items(), key=operator.itemgetter(1))[0])
                print(pred, row["class"])
        except KeyboardInterrupt:
            nlp.to_disk(f"{corpus}-bert")
            for _, row in dataset["test"].iterrows():
                t = nlp(row["text"])
                pred = int(max(t.cats.items(), key=operator.itemgetter(1))[0])
                print(pred, row["class"])
