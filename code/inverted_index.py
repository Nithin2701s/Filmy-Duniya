import pandas as pd
import re
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords



# Load the dataset
data = pd.read_csv('dataset.csv')
documents = data[['id', 'title']]  # Assuming 'id' is the column containing movie IDs

# Preprocess data with stemming and stop words removal
stop_words = set(stopwords.words('english'))


def preprocess(text):
    words = word_tokenize(str(text).lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

preprocessed_documents = [(doc_id, preprocess(doc)) for doc_id, doc in zip(documents['id'], documents['title'])]

# Create an inverted index
inverted_index = {}
for doc_id, doc in preprocessed_documents:
    for token in doc.split():
        if token not in inverted_index:
            inverted_index[token] = [doc_id]
        else:
            inverted_index[token].append(doc_id)

# Convert lists to sets to remove duplicates in document IDs
inverted_index = {token: sorted(list(set(doc_ids))) for token, doc_ids in inverted_index.items()}

# Sort the keys (terms) alphabetically
sorted_inverted_index = dict(sorted(inverted_index.items()))

# Save the sorted inverted index to a JSON file
with open('s_inverted_index.json', 'w') as file:
    json.dump(sorted_inverted_index, file)

print("inverted index created and saved to 'inverted_index.json'.")
