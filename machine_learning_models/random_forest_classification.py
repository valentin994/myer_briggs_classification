from Dataprep import Dataprep
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("../data/mbti_1.csv")
df = df[0:3]
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
df["posts"] = df["posts"].apply(lambda x: Dataprep.base_cleanup(x))
df["posts"] = df["posts"].apply(lambda x: Dataprep.remove_stopwords_tokenize(x, stop_words, lemmatizer))
df = Dataprep.index_encoding(df, "posts")
#df = Dataprep.convert_to_ml_df(df, "encoded", "type")
#ltoi = {l: i for i, l in enumerate(df["type"].unique())}
#df["type"] = df["type"].apply(lambda x: ltoi[x])
#print(df)
# X = df.drop("type", axis=1)
# y = df["type"]
# X_train, y_train, X_test, y_test = train_test_split(X, y, random_state=42)
#
# rfr = RandomForestClassifier()
# rfr.fit(X_train, y_train)
# pred = rfr.predict(X_test)

#print(classification_report(y_test, pred))
