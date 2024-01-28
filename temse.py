import os
import pickle
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from datetime import datetime

# Set your OpenAI API key here
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Instantiate OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Function to load data from a .pkl file
def load_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Function to compute cosine similarity
def calculate_cosine_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Function for embedding-based search
def search_by_embedding(query_embedding, pdf_data_dict, top_n=3):
    similarities = {}
    for pdf_path, (text, embedding) in pdf_data_dict.items():
        similarity = calculate_cosine_similarity(query_embedding, embedding)
        similarities[pdf_path] = similarity

    sorted_paths = sorted(similarities, key=similarities.get, reverse=True)[:top_n]
    return sorted_paths

# Function to get embeddings for a given text
def get_text_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-large")
    return response.data[0].embedding

# Function to generate a response using GPT-3.5 Turbo
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

# Streamlit app layout
def main():
    st.title("Assistent voor de gemeente Temse")

    # System/Instruction prompt
    system_instruction = "Je bent een medewerker van de gemeente Temse. Bij elk antwoord, probeer zo volledig mogelijk te zijn en geef zoveel mogelijk contactinformatie zoals websites, telefoonnummers, namen, etc."

    # Load PDF data
    data_file = 'pdf_data.pkl'
    if not os.path.exists(data_file):
        st.error(f"Data file '{data_file}' not found. Please check the path.")
        return
    pdf_data_dict = load_data(data_file)

    st.write("Stel vragen over temse:")

    # Chat history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # User input
    user_input = st.text_input("Jouw vraag:", key="user_input")

    if st.button("Vraag"):
        with st.spinner('Zoeken in mijn kennis en genereren van een antwoord...'):
            query_embedding = get_text_embedding(user_input)

            # Identify relevant PDFs
            top_pdf_paths = search_by_embedding(query_embedding, pdf_data_dict, top_n=5)

            # Extract text from identified PDFs
            pdf_contents = " ".join([pdf_data_dict[pdf_path][0] for pdf_path in top_pdf_paths])[:5000]

            # Generate a response
            context = f"Gebaseerd op deze documenten: {pdf_contents[:5000]}"
            response = generate_response(user_input, context, system_instruction)

            # Generate a unique key using a timestamp
            unique_key = datetime.now().strftime("%Y%m%d%H%M%S%f")

            # Update chat history
            st.session_state.history.append(("Jij", user_input, unique_key))
            st.session_state.history.append(("AI", response, unique_key))

            # Display chat history
            for role, message, key in st.session_state.history:
                if role == "Jij":
                    st.text_area("", message, key=f"{key}_Jij", height=75)
                else:
                    st.text_area("", message, key=f"{key}_AI", height=150)

if __name__ == "__main__":
    main()
