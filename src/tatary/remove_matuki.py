import pandas as pd
import json

from os import getcwd

DATA_PATH = getcwd() + "/data/dev_inputs.tsv"
ANS_PATH = getcwd() + "/data/dev_outputs.tsv"


from datasets import load_dataset

import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
splits = { 'tt': 'data/tt-00000-of-00001.parquet'}
toxic_lexicon = pd.read_parquet("hf://datasets/textdetox/multilingual_toxic_lexicon/" + splits["tt"])

matuki = toxic_lexicon["text"].tolist()
print(toxic_lexicon.head())

df = pd.read_csv(DATA_PATH, sep="\t")

texts = df["tat_toxic"].tolist()

import pandas as pd
import re
from rapidfuzz import fuzz
import stanza

# -----------------------------
# 0. NLP-пайплайн (Stanza)
# -----------------------------
# скачать модели один раз
# stanza.download("tt")
# nlp = stanza.Pipeline("tt", processors="tokenize,morph,lemma")


# -----------------------------
# 1. Нормализация текста
# -----------------------------
def normalize_text(text):
    text = text.lower()

    # заменяем типичные символы
    text = (
        text.replace("@", "а")
            .replace("0", "о")
            .replace("1", "л")
            .replace("3", "е")
            .replace("*", "")
            .replace("_", "")
            .replace("-", "")
    )

    # убираем многократные буквы: суууука → сука
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    return text


# -----------------------------
# 2. Регэксп на точные словоформы
# -----------------------------
def build_regex(bad_words):
    escaped = [re.escape(w) for w in bad_words]
    pattern = r"\b(" + "|".join(escaped) + r")(?:\w*)\b"
    return re.compile(pattern, flags=re.IGNORECASE)

bad_re = build_regex(matuki)


# -----------------------------
# 3. RapidFuzz-фильтрация одного слова
# -----------------------------
def fuzzy_is_bad(word, bad_list, thr=80):
    for b in bad_list:
        if fuzz.ratio(word, b) >= thr:
            return True
    return False




# -----------------------------
# 5. Общая функция детокса
# -----------------------------
def full_detox(text):
    # normalize
    t=text
    # regex wipe
    t = bad_re.sub("", t)
    t = re.sub(r"\s{2,}", " ", t).strip()


    return t.strip()


# -----------------------------
# 6. Применяем ко всему датасету
# -----------------------------
texts_detoxified = [full_detox(t) for t in texts]
df["tat_detox1"] = texts_detoxified
df.to_csv(ANS_PATH, sep="\t", index=False)
