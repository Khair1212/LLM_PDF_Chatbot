import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def process_pdf():
    # Extract text from PDF using PyMuPDF
    pdf = st.file_uploader("Upload you PDF", type="pdf")
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        return text
    return


## Main Function
def main():
    st.header('PDF CHATBOT ')
    load_dotenv()




    # upload a pdf file (200mb limit)
    #pdf = st.file_uploader("Upload your PDF", type='pdf')
    text = process_pdf()
    st.write(text)


if __name__ == '__main__':
    main()

