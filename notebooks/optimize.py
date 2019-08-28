from pathlib import Path

from flair.datasets import ClassificationCorpus
from flair.embeddings import (
    BertEmbeddings,
    XLNetEmbeddings,
    XLMEmbeddings,
    DocumentRNNEmbeddings,
)
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter
from flair.hyperparameter.param_selection import (
    TextClassifierParamSelector,
    OptimizationValue,
)


if __name__ == "__main__":
    data_folder = Path("classification", "data", "downsampled", "flair")

    for model, embedding in [
        ("bert", BertEmbeddings("bert-base-german-cased")),
        ("xlm", XLMEmbeddings("xlm-mlm-ende-1024")),
        ("xlnet", XLNetEmbeddings()),
    ]:
        for c in ["dramen", "romane", "zeitung", "wikipedia"]:
            if model == "bert" and c != "wikipedia":
                continue
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

            search_space = SearchSpace()
            search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[[embedding]])
            search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[32, 64, 128])
            search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
            search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
            search_space.add(
                Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.1, 0.15, 0.2]
            )
            search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[32])

            param_selector = TextClassifierParamSelector(
                corpus,
                False,
                f"optimization/{model}/{c}",
                "lstm",
                max_epochs=3,
                training_runs=1,
                optimization_value=OptimizationValue.DEV_SCORE,
            )

            param_selector.optimize(search_space, max_evals=5)
