#!/usr/bin/env python
# coding: utf-8

# 참고: https://medium.com/mlearning-ai/build-a-chat-with-csv-app-using-langchain-and-streamlit-94a8b3363aa9
# Chat With Your Document Using Langchain and Streamlit

# Environment Setup

import streamlit as st
import pandas as pd
from pdf2image import convert_from_path
import pytesseract
from PyPDF2 import PdfReader
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
import os
import json
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator


def get_secrets():
    SECRET = {}
    secrets = json.loads(open("./secrets.json").read())
    for k, v in secrets.items():
        SECRET[k] = v
    return SECRET


os.environ['OPENAI_API_KEY'] = get_secrets()['OPENAPI_KEY']
llm = OpenAI(temperature=0.1)
default_data_dir = "./data/"


# Processing Functions
def pdf_to_text(pdf_path):
    # Step 1: Convert PDF to images
    images = convert_from_path(pdf_path)

    with open(default_data_dir + 'output.txt', 'w') as f:  # Open the text file in write mode
        for i, image in enumerate(images):
            # Save pages as images in the pdf
            image_file = default_data_dir + f'page{i}.jpg'
            image.save(image_file, 'JPEG')

            # Step 2: Use OCR to extract text from images
            text = pytesseract.image_to_string(image_file, lang='kor+eng')

            f.write(text + '\n')  # Write the text to the file and add a newline for each page


def load_csv_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.to_csv("uploaded_file.csv")
    return df


def load_txt_data(uploaded_file):
    with open('uploaded_file.txt', 'w') as f:
        f.write(uploaded_file.getvalue().decode())
    return uploaded_file.getvalue().decode()


def load_pdf_data(uploaded_file):
    with open('data/uploaded_file.pdf', 'wb') as f:
        f.write(uploaded_file.getbuffer())
    pdf = PdfReader('data/uploaded_file.pdf')
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    pdf_to_text(default_data_dir + 'uploaded_file.pdf')
    return text


# Main Function


def main():
    st.title("Chat With Your Documents (csv, txt and pdf)")
    file = st.file_uploader("Upload a file", type=["csv", "txt", "pdf"])

    if file is not None:
        if file.type == "text/csv":
            doc = "csv"
            data = load_csv_data(file)
            agent = create_csv_agent(OpenAI(temperature=0), 'uploaded_file.csv', verbose=True)
            st.dataframe(data)

        elif file.type == "text/plain":
            doc = "text"
            data = load_txt_data(file)
            loader = TextLoader('uploaded_file.txt')
            index = VectorstoreIndexCreator().from_loaders([loader])

        elif file.type == "application/pdf":
            doc = "text"
            data = load_pdf_data(file)
            loader = TextLoader('data/output.txt')
            index = VectorstoreIndexCreator().from_loaders([loader])

        # do something with the data

        question = st.text_input("Once uploaded, you can chat with your document. Enter your question here:")
        language = st.text_input("Enter your language")
        submit_button = st.button('Submit')

        if submit_button:
            if doc == "text":
                translation_job = f"""
                문서를 참조하여 다음 문장에 대해 {language}로 자연스럽게 번역해줘.
                {question}
                """
                response = index.query(translation_job)
            else:
                response = agent.run(question)

            if response:
                st.write(response)


if __name__ == "__main__":
    main()
