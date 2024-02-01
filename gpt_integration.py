from openai import OpenAI
import streamlit as st
from utils import get_text_embedding, calculate_cosine_similarity, google_search, scrape_google_results



OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
YOUR_GOOGLE_API_KEY = st.secrets["YOUR_GOOGLE_API_KEY"]
YOUR_GOOGLE_CSE_ID = st.secrets["YOUR_GOOGLE_CSE_ID"]

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_response(prompt, context, system_instruction):
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "system", "content": context},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        max_tokens=1024,
        temperature=0.5,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6
    )
    return response.choices[0].message.content

def translate_text(text, target_language):
    translation_prompt = f"Translate this text to {target_language}: {text}"
    return generate_response(translation_prompt, "", "")

def search_by_embedding(query_embedding, pdf_data_dict, top_n=3):
    similarities = {}
    for pdf_path, (text, embedding) in pdf_data_dict.items():
        similarity = calculate_cosine_similarity(query_embedding, embedding)
        similarities[pdf_path] = similarity
    sorted_paths = sorted(similarities, key=similarities.get, reverse=True)[:top_n]
    return sorted_paths

def is_response_sufficient(response):
    return len(response) > 10000  # Adjust this threshold as needed

def analyze_and_search(user_input, selected_language, response, pdf_contents, system_instruction):
    if not is_response_sufficient(response):
        google_urls = google_search(f"Temse {user_input}", YOUR_GOOGLE_API_KEY, YOUR_GOOGLE_CSE_ID, num_results=5, country='BE')
        google_context = scrape_google_results(google_urls)  # Scrape content from the URLs
        combined_context = f"{pdf_contents}\n\n{google_context}"
        response = generate_response(user_input, combined_context, system_instruction)
    return response