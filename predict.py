import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import pickle
model=""

vectorizer = ""
#with open('SVC_model.pkl', 'rb') as file:
#   model = pickle.load(file)
with open('RF_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
def preprocess_text(text):
    # Tokenization and lowercasing
    tokens = word_tokenize(text.lower())
    
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w in stop_words]
    
    return " ".join(filtered_tokens)
def predict_single_text(text):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)

    # Vectorize the text
    vectorized_text = vectorizer.transform([preprocessed_text])

    # Predict using the trained model
    prediction = model.predict(vectorized_text)

    return prediction[0]

# Test the function with a single text
sample_text = "I hope you can remember what you said to me today, I will let you take it back"
predicted_label = predict_single_text(sample_text)
print("Predicted Label for '{}': {}".format(sample_text, predicted_label))