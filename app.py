import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import time
import uuid, json


class PDFChatbot:
    def __init__(self):
        self.session_id = self.generate_session_id()
        self.vectorstore = None
        self.chain = None
        self.history = []

    @staticmethod
    def generate_session_id():
        return str(uuid.uuid4())

    def process_pdf(self, pdf):
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def text_to_chunks(self, text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        return chunks

    def get_embeddings(self, chunks):
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        return self.vectorstore

    def generate_response(self, query):
        start_time = time.time()
        result = self.chain(
            {"question": query, 'chat_history': self.history}, return_only_outputs=True)
        end_time = time.time()
        response_time = end_time - start_time
        return result["answer"], response_time

    def text_to_json(self, text):
        try:
            # Assuming text is in a valid JSON format
            json_data = json.loads(text)
            return json_data
        except json.JSONDecodeError as e:
            # If text is not in JSON format, handle the exception
            st.error(f"Error decoding JSON: {e}")
            return None
    def run_chatbot(self):
        st.header('PDF CHATBOT ')
        load_dotenv()

        # upload a pdf file and extract text (200mb limit)
        pdf = st.file_uploader("Upload your PDF", type="pdf")
        query = st.text_input("Enter a question:", "")

        if pdf is not None:
            text = self.process_pdf(pdf)
            chunks = self.text_to_chunks(text)
            self.get_embeddings(chunks)

            memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
            self.chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=0.3),
                retriever=self.vectorstore.as_retriever(),
                memory=memory
            )

            # Interaction with the user
            if query:
                # Generate a response based on the user's question and the chat history
                response, response_time = self.generate_response(query)

                # Display the question and response
                st.write(f"User: {query}")
                # st.write(f"ChatBot: {response}")

                # json
                json_data = {"response": response}

                # Display JSON data in Streamlit
                st.write("Chatbot:")
                st.json(json_data)
                st.write(f"Response Time: {response_time}")

                # Update the chat history
                self.history.append({"role": "user", "content": query})
                self.history.append({"role": "assistant", "content": response})


if __name__ == '__main__':
    chatbot = PDFChatbot()
    chatbot.run_chatbot()