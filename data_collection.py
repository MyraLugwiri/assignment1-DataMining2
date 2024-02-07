# importing necessary libraries for the webscrapping
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import pandas as pd
import numpy as np
import string
import nltk
import joblib
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')
from nltk import word_tokenize

nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
import inflect


def scrape_articles(url):
    """
    crawls the web url and returns the news articles, links to the articles
    and the associated article titles
     """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # list to store the headline
    titles = []
    # looping through all h2 to get the article titles
    for title in soup.find_all('span', {'class': 'container__headline-text'}):
        titles.append(title.text.strip())

    # list to store the links to the full articles
    unique_links = set()
    # links = []
    titles_link = soup.find_all('a', {'class': 'container__link'})
    for link in titles_link:
        # Access the 'href' attribute of each anchor tag
        href = link.get('href')
        if not href.startswith('http'):
            href = url + href
        # links.append(href)
        if not is_video_link(href):
            # links.append(href)
            unique_links.add(href)
    links = list(unique_links)

    # list to store the article content
    content_list = []
    for link in links:
        # making a request to the article specific page
        content_response = requests.get(link, stream=True, timeout=(20, 50))
        content_soup = BeautifulSoup(content_response.text, 'html.parser')

        # extracting content from the article page
        content = content_soup.find_all('p', {'class': 'paragraph'})
        full_content = ' '.join([paragraph.text.strip() for paragraph in content])
        content_list.append(full_content)

    # Check and ensure that all lists have the same length
    min_length = min(len(titles), len(links), len(content_list))
    titles = titles[:min_length]
    links = links[:min_length]
    content_list = content_list[:min_length]

    return titles, links, content_list


def is_video_link(link):
    """
    Checks if the link might contain video content based on common URL patterns.
    """
    video_keywords = ['videos', 'watch', 'interactive', 'video']
    return any(keyword in link.lower() for keyword in video_keywords)


# pre-processing the articles' text data
def preprocessing_articles(text):
    # Check if the element is a list, and join it into a string
    if isinstance(text, list):
        text = ' '.join(map(str, text))

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    punctuation_set = set(string.punctuation)
    text = ''.join(char for char in text if char not in punctuation_set)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Lemmatization
    lemma = WordNetLemmatizer()
    lemmat = [lemma.lemmatize(token) for token in tokens]

    # Converting numerical values to words
    inflect_en = inflect.engine()
    text_words = [inflect_en.number_to_words(token) if token.isdigit() else token for token in lemmat]

    # Processed text as a single string
    processed_text = ' '.join(text_words)

    return processed_text


def main():
    # collecting the data from the CNN web page
    url = 'https://edition.cnn.com'
    article_titles, article_links, full_article = scrape_articles(url)

    # creating a dataframe using the data collected from web scrapping
    data = {'title': article_titles, 'content': full_article, 'link': article_links}
    collected_articles = pd.DataFrame(data=data)

    # preprocessing the news articles
    collected_articles['processed_articles'] = collected_articles['content'].apply(preprocessing_articles)

    # performing data transformation using TFIDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(collected_articles['processed_articles'])

    # performing clustering using k-means algorithm
    kmeans = KMeans(n_clusters=10, random_state=42)
    collected_articles['cluster'] = kmeans.fit_predict(tfidf_matrix)

    # collected_articles.to_csv('/collected_articles.csv')


if __name__ == '__main__':
    main()
