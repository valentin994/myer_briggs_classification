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
        text = text.split()
        return text

    @staticmethod
    def remove_stopwords_tokenize(text: list, stop_words: list, lemmatizer):
        """
        :paramtext:Texttobecleaned
        :return:Cleanedtext,withoutlinks,numbersandpunctuation
        """
        text = [word for word in text if word not in stop_words]
        text = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(text)
        return text

    @staticmethod
    def index_encoding(df: pd.DataFrame, text_column: str):
        def spread_array(values, max_len):
            encoded = np.zeros(max_len, dtype=int)
            encoded[:len(values)] = values
            return encoded

        counts = Counter()
        max_len: int = 0
        for index, row in df.iterrows():
            if len(row[text_column]) > max_len:
                max_len = len(row[text_column])
            counts.update(row['posts'])
        for word in list(counts):
            if counts[word] < 2:
                del counts[word]
        vocab2index = {"": 0, "UNK": 1}
        words = ["", "UNK"]
        for word in counts:
            vocab2index[word] = len(words)
            words.append(word)
        # encoded = np.zeros(max_len, dtype=int)
        df["encoded"] = df[text_column].apply(
            lambda x: np.array([vocab2index.get(word, vocab2index["UNK"]) for word in x]))
        df["encoded"] = df["encoded"].apply(lambda x: spread_array(x, max_len))
        return df

    @staticmethod
    def convert_to_ml_df(df: pd.DataFrame, text: str, type: str):
        column_len = len(df[text][0])
        column_names = {}
        for i in range(column_len):
            column_names[f"Word {i}"] = []
        for index, row in df.iterrows():
            for data in range(column_len):
                column_names[f"Word {data}"].append(row[text][data])
        new_df = pd.DataFrame(column_names)
        new_df[type] = df[type]
        return new_df
    # TODO Bag of Words
    # TODO TF-IDF Encoding
    # TODO Word2Vec
    # TODO BERT Encoding

# df = pd.read_csv("./data/mbti_1.csv")
# df = df[0:10]
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()
# df["posts"] = df["posts"].apply(lambda x: Dataprep.base_cleanup(x))
# df["posts"] = df["posts"].apply(lambda x: Dataprep.remove_stopwords_tokenize(x, stop_words, lemmatizer))
# df = Dataprep.index_encoding(df, "posts")
