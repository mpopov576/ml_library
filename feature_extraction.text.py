import numpy as np
import re
from collections import Counter

class TfidfVectorizer:
    def __init__(self, max_features=None, max_df=None, min_df=None, stop_words=None, lowercase=True):
        self.stop_words = stop_words
        self.max_features = max_features
        self.max_df = max_df if max_df is not None else 1.0
        self.min_df = min_df if min_df is not None else 1
        self.stop_words = set(stop_words) if stop_words else None
        self.lowercase = lowercase

        self.vocabulary_ = {}
        self.idf_ = None

    def _tokenize(self, doc):
        if self.lowercase:
            doc = doc.lower()

        tokens = re.findall(r"\b\w+\b", doc)

        if self.stop_words:
            tokens = [t for t in tokens if t not in self.stop_words]

        return tokens

    def fit(self, corpus):
        n_docs = len(corpus)
        df = {}

        for doc in corpus:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                df[token] = df.get(token, 0) + 1

        tokens = []
        for token, freq in df.items():
            if isinstance(self.min_df, float):
                min_ok = freq >= self.min_df * n_docs

            else: min_ok = freq >= self.min_df

            if isinstance(self.max_df, float):
                max_ok = freq <= self.max_df * n_docs

            else: max_ok = freq <= self.max_df

            if min_ok and max_ok:
                tokens.append((token, freq))

        tokens.sort(key=lambda x: x[1], reverse=True)

        if self.max_features:
            tokens = tokens[:self.max_features]

        self.vocabulary_ = {token: i for i, (token, _) in enumerate(tokens)}

        self.idf_ = np.zeros(len(self.vocabulary_))
        for token, i in self.vocabulary_.items():
            self.idf_[i] = np.log((1 + n_docs) / (1 + df[token])) + 1

        return self

    def transform(self, corpus):
        if not self.vocabulary_:
            raise ValueError("Call fit first")

        X = np.zeros((len(corpus), len(self.vocabulary_)))

        for i, doc in enumerate(corpus):
            tokens = self._tokenize(doc)
            tf = Counter(tokens)

            for token, count in tf.items():
                if token in self.vocabulary_:
                    j = self.vocabulary_[token]
                    X[i, j] = count

            X[i] *= self.idf_

            norm = np.linalg.norm(X[i])
            if norm > 0:
                X[i] /= norm

        return X

    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)
