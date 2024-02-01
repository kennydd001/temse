import streamlit as st
import logging
from gpt_integration import generate_response, translate_text, get_text_embedding, search_by_embedding, analyze_and_search
from utils import load_data
import os

# Basic configuration for logging
logging.basicConfig(level=logging.INFO)

def setup_app():
    languages = {
        'Nederlands': 'Dutch', 'Français': 'French', 'Deutsch': 'German',
        'Español': 'Spanish', 'Italiano': 'Italian', 'العربية': 'Arabic',
        'Polski': 'Polish', 'Türkçe': 'Turkish', 'English': 'English',
        'Українська': 'Ukrainian', 'Русский': 'Russian'
    }
    selected_language = st.selectbox("Select Language", list(languages.keys()))
    system_instruction = "Je bent een medewerker van de gemeente Temse. Bij elk antwoord, probeer zo volledig mogelijk te zijn en geef zoveel mogelijk contactinformatie zoals websites, telefoonnummers, namen, etc."
    return system_instruction, selected_language, languages

def load_pdf_data():
    data_file = 'pdf_data.pkl'
    if not os.path.exists(data_file):
        st.error(f"Data file '{data_file}' not found. Please check the path.")
        return None
    return load_data(data_file)

def main():
    st.title("Assistent voor de gemeente Temse / Assistent for the municipality of Temse")
    system_instruction, selected_language, languages = setup_app()

    pdf_data_dict = load_pdf_data()
    if pdf_data_dict is None:
        return

    st.write("Stel vragen over Temse:")
    user_input = st.text_input("Jouw vraag: / your question", key="user_input")

    if st.button("Vraag / Ask"):
        logging.info(f"User input: {user_input}")
        with st.spinner('Zoeken in mijn kennis en genereren van een antwoord...'):
            # Translate user input to English if it's not already in English
            translated_input = translate_text(user_input, 'English') if selected_language != 'English' else user_input
            query_embedding = get_text_embedding(translated_input)
            top_pdf_paths = search_by_embedding(query_embedding, pdf_data_dict, top_n=5)

            pdf_contents = " ".join([pdf_data_dict[pdf_path][0] for pdf_path in top_pdf_paths])[:5000]
            context = f"Gebaseerd op deze documenten: {pdf_contents[:5000]}"
            response = generate_response(translated_input, context, system_instruction)

            final_response = analyze_and_search(user_input, selected_language, response, pdf_contents, system_instruction)
            # Translate the response back only if the user's selected language is not English
            translated_response = translate_text(final_response, languages[selected_language]) if selected_language != 'English' else final_response

            st.text_area("Antwoord", translated_response, height=150)

if __name__ == "__main__":
    main()
