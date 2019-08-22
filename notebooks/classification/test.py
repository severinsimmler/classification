scores = list()

for corpus in {"wikipedia", "dramen", "romane", "wikipedia", "zeitung"}:
    score = dict(classification.classic_pipeline(corpus, downsample=False))
    score = pd.Series(score)
    score.name = corpus
    scores.append(score)
