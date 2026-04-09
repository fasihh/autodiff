import json
import pickle
import unicodedata
import numpy as np
from fygrad.module import Module, Embedding
from fygrad.node import Node


class Tokenizer:
    def __init__(self, vocab_file: str):
        with open(vocab_file, encoding="utf-8") as f:
            data = json.load(f)
        # vocab is token -> id
        self.vocab = data["model"]["vocab"]
        self.vocab_set = set(self.vocab.keys())

    def normalize(self, text):
        text = text.lower()
        text = unicodedata.normalize("NFD", text)
        return "".join(c for c in text if unicodedata.category(c) != "Mn")

    def split_words(self, text):
        tokens, current = [], ""
        for char in text:
            if char.isspace():
                if current:
                    tokens.append(current)
                    current = ""
            elif not char.isalnum():
                if current:
                    tokens.append(current)
                    current = ""
                tokens.append(char)
            else:
                current += char
        if current:
            tokens.append(current)
        return tokens

    def wordpiece(self, word):
        if word in self.vocab_set:
            return [word]
        tokens, start = [], 0
        while start < len(word):
            end, best = len(word), None
            while start < end:
                sub = ("##" if start > 0 else "") + word[start:end]
                if sub in self.vocab_set:
                    best = sub
                    break
                end -= 1
            if best is None:
                return ["[UNK]"]
            tokens.append(best)
            start = end
        return tokens

    def encode(self, text):
        text = self.normalize(text)
        words = self.split_words(text)
        subwords = ["[CLS]"]
        for w in words:
            subwords.extend(self.wordpiece(w))
        subwords.append("[SEP]")
        ids = [self.vocab.get(t, self.vocab["[UNK]"]) for t in subwords]
        return subwords, ids


class InputEmbedding(Module):
    def __init__(self, emb_size: int, v_dim: int = 768):
        super().__init__()

        self.word_embeddings = Embedding((emb_size, v_dim), "word_embeddings", device=self.device)

    def forward(self, indices: np.ndarray):
        return self.word_embeddings(indices)


def toids(sent: str):
    tok = Tokenizer("test/tokenizer.json")
    tokens, ids = tok.encode(sent)
    print(tokens)
    return ids

def cosine_sim(a: np.ndarray, b: np.ndarray):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

with open("test/weights.pkl", "rb") as f:
    bert_state = pickle.load(f)

print(f"{len(bert_state)} parameters loaded")

key = "embeddings.word_embeddings.weight"
emb_weights: np.ndarray = bert_state[key].numpy()

embeddings = InputEmbedding(*emb_weights.shape)
embeddings.load_state_dict({
    "word_embeddings": {
        "emb": {
            "value": emb_weights,
            "shape": emb_weights.shape,
            "device": "cpu"
        }
    }
})

print(f"{key} loaded to fygrad 'InputEmbedding'")

out = embeddings(np.array(toids("man king woman queen")))
print("king - man + woman = queen", cosine_sim(out.value[2] - out.value[1] + out.value[3], out.value[4]))
print("queen - woman + man = king", cosine_sim(out.value[4] - out.value[3] + out.value[1], out.value[2]))

out = embeddings(np.array(toids("hitler germany italy mussolini")))
print("hitler - germany + italy = mussolini", cosine_sim(out.value[1] - out.value[2] + out.value[3], out.value[4]))

out = embeddings(np.array(toids("nazi hitler communism oranges")))
print("hitler = nazi", cosine_sim(out.value[1], out.value[2]))
print("hitler = communism", cosine_sim(out.value[1], out.value[3]))
print("hitler = orange", cosine_sim(out.value[1], out.value[4]))
