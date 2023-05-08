import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load NLTK stop words and stemmer
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load the CSV file
df = pd.read_csv('dialogs.csv', header=None, names=['context', 'response', 'intent'])

# Preprocess the text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return " ".join(stemmed_tokens)

df['context'] = df['context'].apply(preprocess_text)
df['response'] = df['response'].apply(preprocess_text)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer on the preprocessed text
vectorizer.fit(df['context'].values.tolist() + df['response'].values.tolist())

# Create TF-IDF vectors for the context and response
context_vectors = vectorizer.transform(df['context'])
response_vectors = vectorizer.transform(df['response'])

# Add the TF-IDF vectors to the DataFrame
for i in range(context_vectors.shape[1]):
    df['context_tfidf_' + str(i)] = context_vectors[:, i].toarray()
for i in range(response_vectors.shape[1]):
    df['response_tfidf_' + str(i)] = response_vectors[:, i].toarray()

# Print the preprocessed DataFrame
print(df.head())