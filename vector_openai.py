import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# Set your OpenAI API key here
OPENAI_API_KEY = "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"

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
def generate_response(prompt, context, system_prompt=""):
    messages = [
        {"role": "system", "content": system_prompt},  # System/Instruction prompt
        {"role": "system", "content": context},        # Context of the conversation
        {"role": "user", "content": prompt}            # User's actual query
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

# Main function for command line usage
def main():
    data_file = 'pdf_data.pkl'
    if not os.path.exists(data_file):
        print(f"Data file '{data_file}' not found. Please check the path.")
        return
    pdf_data_dict = load_data(data_file)

    print("Welcome to a conversation with AI! Ask questions about the content of the PDFs.")

    # System/Instruction prompt
    system_instruction = "Je bent een mederwerker van de gemeente temse, bij elk antwoord probeer je zo volledig mogelijk te zijn en zoveel mogelijk contact info zoals websites, telefoon nummers, namen, etc me te geven."

    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            break

        query_embedding = get_text_embedding(user_query)

        # Identify relevant PDFs
        top_pdf_paths = search_by_embedding(query_embedding, pdf_data_dict, top_n=5)
        print("Selected PDFs:", top_pdf_paths)

        # Extract text from identified PDFs
        pdf_contents = " ".join([pdf_data_dict[pdf_path][0] for pdf_path in top_pdf_paths])[:5000]

        # Generate a response with system instructions
        context = f"Based on these documents: {pdf_contents[:5000]}"
        response = generate_response(user_query, context, system_instruction)

        print("AI:", response)

if __name__ == "__main__":
    main()
