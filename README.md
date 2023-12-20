### Movie Recommender System:

## Description
This project aims to develop a movie retrieval system using natural language processing techniques. 
It leverages an inverted index to enable efficient and relevant movie title search based on user queries.
It takes relevant feedback from the use and remove non-relevant movies
from the results.

The project is divided into five main components:
1.Data collection
> 	Drive link for Dataset.
(https://drive.google.com/file/d/1Gzc0dxW4cAHvXCnPYzeDlYgb-XgiZuXu/view?usp=sharing)
>Save dataset to code directory

2. Inverted Index:
> This component is responsible for creating an inverted index of terms in movie titles and their corresponding movie IDs.

3.Tf-idf matrix :
> Creating Tf-idf matrix for movie titles containing query terms and ranking them based on cosine similarity

4.Recommendation System UI:
>UI using flask in python for showing ranked movies and getting relevance feedback

5.Relevance Feedbak and P-R curve
>Precision-Recall curve based on user feedback

## Installation:

To install the required libraries, you can use the following pip command:
pip install scikit-learn
pip install flask 
pip install nltk
pip install pandas
pip install matplotlib

## After installing nltk
Run python iterpreter using command :python or py
commands needed to download stopwords
$ import nltk
$ nltk.download('stopwords')

## Run the code 
$ py inverted index.py (takes few seconds to create inverted_index.json)
$ py app.py (takes few seconds to load inverted_index and dataset. Please wait until debugger becomes (active shown in terminal))

## Open the browser and type the url
After the app starts running you find an url in the terminal.
Copy that url and paste in the browser.

## The app will show the home page.
Searh for a movie title:
Example queries : 
> American psycho
> Dasara

## You find the movies retrived and ranked
> For overview and other details click on view details

## Relevance feedback
> If you find a movie not relevant click on not relevant
> If you find a movie relevant click on relevant

## The app will update the results with the feedback and remove the non-relevant movies and retrive the movies that are not retrieved in the first process and based on relevance feedback, Precision-Recall curve is seved in pr_curve

