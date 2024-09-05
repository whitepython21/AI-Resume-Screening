import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    return ' '.join([word for word in tokens if word not in stop_words])

def vectorize_text(text_data):
    tfidf = TfidfVectorizer(max_features=1000)
    return tfidf.fit_transform(text_data).toarray()
