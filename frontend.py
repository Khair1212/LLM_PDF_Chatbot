import streamlit as st
from dotenv import load_dotenv
import uuid
import requests

class PDFChatbotApp:
    def __init__(self):
        st.header('PDF CHATBOT ')
        load_dotenv()
        self.session_id = self.generate_session_id()
        self.pdf = None
        self.query = None

    @staticmethod
    def generate_session_id():
        return str(uuid.uuid4())

    def upload_pdf(self):
        self.pdf = st.file_uploader("Upload your PDF", type="pdf")

    def get_user_input(self):
        self.query = st.text_input("Enter a question:", "")

    def process_uploaded_pdf(self):
        if self.pdf is not None:
            with open(f"/tmp/{self.pdf.name[:-4]}.pdf", "wb") as f:
                f.write(self.pdf.read())

    def interact_with_chatbot(self):
        if self.query:
            response = self.get_chatbot_response()
            st.write(f"Chatbot: {response}")

    def get_chatbot_response(self):
        url = f"http://localhost:5000/get_chatbot_response/{self.session_id}{self.pdf.name[:-4]}{self.query}"
        response = requests.post(url, json={"query": self.query}, headers={"Content-Type": "application/json"})
        return response.json()['response']

    def run(self):
        self.upload_pdf()
        self.get_user_input()
        self.process_uploaded_pdf()
        self.interact_with_chatbot()

if __name__ == '__main__':
    pdf_chatbot_app = PDFChatbotApp()
    pdf_chatbot_app.run()

