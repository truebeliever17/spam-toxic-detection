import torch
import string
import gensim.downloader as api
import numpy as np
from torch import nn
from nltk.tokenize import WordPunctTokenizer


class SpamToxicDetector:
    def __init__(self, model, tokenizer, word2vec, idx2label):
        self.model = model
        self.tokenizer = tokenizer
        self.idx2label = idx2label
        self.word2vec = word2vec
        self.mean = np.mean(word2vec.vectors, axis=0)
        self.std = np.std(word2vec.vectors, axis=0)

    def get_tokens(self, text):
        return [token for token in self.tokenizer.tokenize(text.lower()) if
                all(symbol not in string.punctuation for symbol in token) and len(token) >= 3]

    def get_avg_embedding(self, tokens):
        embedding = [(self.word2vec[token] - self.mean) / self.std for token in tokens if token in self.word2vec]

        if len(embedding) == 0:
            embedding = np.zeros(self.word2vec.vector_size)
        else:
            embedding = np.mean(embedding, axis=0)
        return embedding

    def make_prediction(self, text):
        tokens = self.get_tokens(text)
        embedding = self.get_avg_embedding(tokens)
        pred = self.model(torch.tensor(embedding).float())
        pred_label_idx = torch.argmax(pred).item()
        return self.idx2label[pred_label_idx]


def load_w2v():
    word2vec = api.load("glove-twitter-200")
    return word2vec


def load_tokenizer():
    return WordPunctTokenizer()


def load_model(path, embed_size, num_classes):
    model = nn.Sequential(
        nn.Linear(embed_size, 128),
        nn.ReLU(),
        nn.Linear(128, 16),
        nn.ReLU(),
        nn.Linear(16, num_classes)
    )
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model
