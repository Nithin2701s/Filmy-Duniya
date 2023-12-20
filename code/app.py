import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import matplotlib

app = Flask(__name__)
matplotlib.use('Agg')


# Load the dataset
data = pd.read_csv('dataset.csv')
documents = data[['id', 'title']]  # Assuming 'id' is the column containing document IDs

# Store the initial search results
initial_results = []
ground_truth_relevance=[]
n=1

# Load the inverted index from JSON file
with open('inverted_index.json', 'r') as file:
    inverted_index = json.load(file)

# Preprocess the dataset
stop_words = set(stopwords.words('english'))
documents.loc[:, 'title'] = documents['title'].apply(
    lambda x: ' '.join([str(word).lower() for word in word_tokenize(str(x)) if word.isalpha() and word not in stop_words])
)

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents['title'])



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    global initial_results
    query = request.json.get('query')

    if query:
        # Preprocess the query
        preprocessed_query = ' '.join([word.lower() for word in word_tokenize(query) if word.isalpha()])

        # Calculate cosine similarity
        query_vector = vectorizer.transform([preprocessed_query])
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)

        ranked_documents = []
        for doc_id, title, similarity in zip(documents['id'], documents['title'], cosine_similarities[0]):
            if similarity > 0:
                ranked_documents.append([doc_id, title, similarity])
        
        # Initialize ground truth relevance        
        
        # Sort the ranked_documents based on similarity in descending order
        ranked_documents.sort(key=lambda x: x[2], reverse=True)

        # Store the initial search results
        initial_results = ranked_documents

        # Return top 10 results to the template
        top_results = initial_results[:15]  
        final_results = []
        i = 1
        for result in top_results:
            doc = data[data['id'] == result[0]]
            doc.to_dict()
            overview = ''
            if doc.overview.isna().any():
                overview = 'no overview'
            else:
                overview = doc.overview.iloc[0]  
            if not(doc.genres.isna().any()) and not(doc.vote_average.isna().any()) and not(doc.runtime.isna().any()) and not(doc.credits.isna().any()):    
             final_results.append([result[0], doc.title.iloc[0], overview,doc.genres.iloc[0],doc.vote_average.iloc[0], doc.runtime.iloc[0],doc.credits.iloc[0],result[2]])

        return jsonify({'results': final_results})
    else:
        return jsonify({'results': []})

def pr_curve():
    global initial_results, ground_truth_relevance, n

    # Extract relevance scores and ground truth for PR curve
    relevance_scores = [result[2] for result in initial_results[:30]]
    y_true = ground_truth_relevance

    # Calculate precision and recall
    i=1
    precision=[]
    recall=[]
    for relevance in ground_truth_relevance:
            recall.append(sum( trues for trues in y_true[:i])/len(y_true))
            precision.append(sum( trues for trues in y_true[:i])/len(y_true[:i]))
            i+=1
    print(precision)
    print(recall)        

    # Plot the PR curve
    plt.plot(recall, precision, label=f'PR Curve {n}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    # Save the PR curve plot to a file
    curve_filename = 'pr_curve.png'
    plt.savefig(curve_filename)
    n+=1
@app.route('/feedback', methods=['POST'])
def feedback():
    global initial_results, ground_truth_relevance
    feedback_data = request.json.get('feedback')
    
     # Update ground truth relevance based on feedback
    for feedback_item in feedback_data:
        document_id = feedback_item['document_id']
        relevance = feedback_item['relevance']
        ground_truth_relevance.append(1 if relevance == 'relevant' else 0)
    pr_curve()
    ground_truth_relevance=[]
    # Assuming feedback_data is a list of dictionaries containing document_id and relevance
    for feedback_item in feedback_data:
        document_id = feedback_item['document_id']
        relevance = feedback_item['relevance']

        # Update the initial_results based on feedback
        for result in initial_results:
            if result[0] == document_id and relevance=='not-relevant':
                initial_results.remove(result)

        # Sort the initial_results based on the updated relevance scores
        initial_results.sort(key=lambda x: x[2], reverse=True)

    # Return the updated results to the template
    updated_results = initial_results[:15]
    final_results = []
    for result in updated_results:
        doc = data[data['id'] == result[0]]
        doc.to_dict()
        overview = ''
        if doc.overview.isna().any():
            overview = 'no overview'
        else:
            overview = doc.overview.iloc[0]
        if not(doc.genres.isna().any()) and not(doc.vote_average.isna().any()) and not(doc.runtime.isna().any()) and not(doc.credits.isna().any()):             
         final_results.append([result[0], doc.title.iloc[0], overview,doc.genres.iloc[0],doc.vote_average.iloc[0], doc.runtime.iloc[0],doc.credits.iloc[0],result[2]])


    return jsonify({'results': final_results})

if __name__ == '__main__':
    app.run(debug=True)
