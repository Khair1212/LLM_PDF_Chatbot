import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def process_pdf():
    # Extract text from PDF using
    pdf = st.file_uploader("Upload you PDF", type="pdf")
    pdf_reader = PdfReader(pdf)

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

    for page in pdf_reader.pages:
        text += page.extract_text()

def text_to_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return chunks

def get_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

## Main Function
def main():
    st.header('PDF CHATBOT ')
    load_dotenv()
    # upload a pdf file and extract text (200mb limit)
    text = process_pdf()
    chunks = text_to_chunks(text)
    vectorstore = get_embeddings(chunks)
    st.write(chunks)


if __name__ == '__main__':
    main()
