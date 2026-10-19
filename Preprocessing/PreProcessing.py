"""Prepare text columns of a DataFrame for clustering (TF-IDF)."""
import re
import unicodedata
from PreprocessingClasses import TextPreprocessing
import ftfy
import pandas as pd



def normalize_text(text, lowercase=True):
    text = ftfy.fix_text(str(text))
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lowercase else text

if __name__ == "__main__":
    from sklearn.cluster import KMeans

    data = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "review": [
            "Great battery life and screen",
            "Battery   lasts all day, great screen",
            "   ",
            "Terrible customer service, slow shipping",
            "Shipping was slow and support was terrible",
            None,
        ],
    })
    prep = TextPreprocessing(tfidf_params={"min_df": 1})
    X = prep.fit_transform(data)
    print("text columns:", prep.text_columns_, "| shape:", X.shape)
    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
    print(dict(zip(prep.index_, labels)))