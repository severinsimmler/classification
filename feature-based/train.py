from pathlib import Path

from flair.datasets import ClassificationCorpus
from flair.embeddings import BertEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer


if __name__ == "__main__":
    data_folder = Path("..", "classification", "data", "downsampled", "flair")
    for c in ["dramen", "romane", "zeitung", "wikipedia"]:
        if c == "dramen":
            dropout = 0.4949748569324362
            hidden_size = 64
            learning_rate = 0.05
            mini_batch_size = 32
            rnn_layers = 2
            hidden_size = 64
        elif c == "romane":
            dropout = 0.3874021064249358
            hidden_size = 128
            learning_rate = 0.1
            mini_batch_size = 32
            rnn_layers = 2
        elif c == "zeitung":
            dropout = 0.08010071245223199
            hidden_size = 32
            learning_rate = 0.05
            mini_batch_size = 32
            rnn_layers = 1
        elif c == "wikipedia":
            dropout = 0.3414306905512738
            hidden_size = 128
            learning_rate = 0.05
            mini_batch_size = 32
            rnn_layers = 2

        test_file = f"{c}-downsampled-test-flair.txt"
        dev_file = f"{c}-downsampled-val-flair.txt"
        train_file = f"{c}-downsampled-train-flair.txt"

        corpus = ClassificationCorpus(
            data_folder,
            test_file=test_file,
            dev_file=dev_file,
            train_file=train_file,
        )

        label_dict = corpus.make_label_dictionary()

        document_embeddings = DocumentRNNEmbeddings(
            BertEmbeddings("bert-base-german-cased"),
            hidden_size=hidden_size,
            dropout=dropout,
            rnn_layers=rnn_layers,
        )
        classifier = TextClassifier(
            document_embeddings, label_dictionary=label_dict
        )

        trainer = ModelTrainer(classifier, corpus)

        trainer.train(
            f"models/{c}",
            learning_rate=learning_rate,
            mini_batch_size=mini_batch_size,
            max_epochs=100,
        )

