import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

class Dataprep:
    @staticmethod
    def base_cleanup(text: str):
        """
        :paramtext:Texttolemmatize
        :paramstop_words:Arraywithstopwordstoremovefromtext
        :paramlemmatizer:nltk.stemWordNetLemmatizer
        :return:Cleanedtext
        """
        text = re.sub(r"http\S+", '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'[0-9]', "", text)
        text = text.replace("_", "")
        text = text.lower()
        return text

    @staticmethod
    def remove_stopwords_tokenize(text: list, stop_words: list, lemmatizer):
        """
        :paramtext:Texttobecleaned
        :return:Cleanedtext,withoutlinks,numbersandpunctuation
        """
        text = text.split()
        text = [word for word in text if word not in stop_words]
        text = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(text)
        return text

    @staticmethod
    def index_encoding():
        pass

    # TODO Bag of Words
    # TODO TF-IDF Encoding
    # TODO Word2Vec
    # TODO BERT Encoding

df = pd.read_csv("./data/cleaned_mbti.csv")
df = df.dropna()
df["posts"] = df["posts"].apply(lambda x: x.split())
counts = Counter()
for index, row in df.iterrows():
    counts.update(row['posts'])
