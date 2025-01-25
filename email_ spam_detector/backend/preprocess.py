import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    # Remove URLs, HTML tags, special characters, and numbers
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetical characters
    text = text.lower()  # Convert to lowercase
    return text

# Function to apply stemming and remove stopwords
def preprocess_text(text):
    text = clean_text(text)
    words = word_tokenize(text)
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Feature engineering: Create a TF-IDF vectorizer
def extract_features(df):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['text']).toarray()
    return X, vectorizer

# Load and preprocess the dataset
def load_data(file_path):
    df = pd.read_csv(file_path, encoding='latin-1')
    df.columns = ['label', 'text']
    df['text'] = df['text'].apply(preprocess_text)
    return df
