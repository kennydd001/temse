import requests
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import streamlit as st
import logging
from bs4 import BeautifulSoup

def load_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def calculate_cosine_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

def google_search(query, api_key, cse_id, num_results=10, country='BE'):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'cx': cse_id,
        'key': api_key,
        'num': num_results,  # Make sure this parameter is correctly named and used
        'cr': f'country{country}'  # Restricting search to Belgium
    }
    response = requests.get(search_url, params=params)
    search_results = response.json()
    return [item['link'] for item in search_results.get('items', [])]

def scrape_google_results(urls):
    scraped_data = ""
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            texts = soup.find_all('p')  # Adjust tag as needed
            for text in texts:
                scraped_data += f"{text.get_text()} "
        except Exception as e:
            logging.error(f"Error scraping URL {url}: {e}")
    return scraped_data
def get_text_embedding(text):
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(input=text, model="text-embedding-3-large")
    return response.data[0].embedding
