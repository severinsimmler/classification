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

MAX_EPOCHS = 50
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


if __name__ == "__main__":
    for model in ["de_pytt_bertbasecased_lg", "xlm-mlm-ende-1024"]:
        for corpus in ["dramen", "romane", "zeitung", "wikipedia"]:
            dataset = preprocessing.load(corpus, split=True, downsample=True)
            if model == "de_pytt_bertbasecased_lg":
                nlp = spacy.load(model)
            else:
                nlp = PyTT_Language(pytt_name=model, meta={"lang": "de"})
                nlp.add_pipe(nlp.create_pipe("sentencizer"))
                nlp.add_pipe(PyTT_WordPiecer.from_pretrained(nlp.vocab, model))
                nlp.add_pipe(PyTT_TokenVectorEncoder.from_pretrained(nlp.vocab, model))
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
            for epoch in range(MAX_EPOCHS):
                logging.error(f"Epoch #{epoch + 1}")
                random.shuffle(train_data)
                batches = spacy.util.minibatch(train_data, size=spacy.util.compounding(1., 32., 1.001))
                for batch in batches:
                    optimizer.pytt_lr = next(learn_rates)
                    texts, annotations = zip(*batch)
                    dropout = next(DROPOUT)
                    nlp.update(
                        texts, annotations, sgd=optimizer, drop=dropout, losses=losses
                    )
                stats = list()
                for _, row in dataset["val"].iterrows():
                    t = nlp(row["text"])
                    pred = int(max(t.cats.items(), key=operator.itemgetter(1))[0])
                    if pred == row["class"]:
                        stats.append(1)
                    else:
                        stats.append(0)
                SCORES["DEV"].append(sum(stats) / len(dataset["val"]))

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

            with open(f"{model}-{corpus}.txt", "w", encoding="utf-8") as f:
                f.write(json.dumps(SCORES, indent=2))
