import json
import time
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

def get_pdf_content(pdf_name):
    # Implement logic to access and extract text from the saved PDF based on session_id
    with open(f"{pdf_name}.pdf", "rb") as f:
        pdf_reader = PdfReader(f)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def text_to_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return chunks

def get_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def generate_response(chain, history, query):
    start_time = time.time()  # Record the start time
    result = chain(
        {"question": query, 'chat_history': history}, return_only_outputs=True)
    end_time = time.time()  # Record the end time
    response_time = end_time - start_time
    return result["answer"], response_time

def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.route('/get_chatbot_response/<session_id>/<pdf_name>/<query>', methods=['POST'])
    def get_chatbot_response(session_id, pdf_name, query):
        query_question = query
        print(query_question)

        text = get_pdf_content(pdf_name)
        chunks = text_to_chunks(text)
        vectorstore = get_embeddings(chunks)

        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.3),
                                                      retriever=vectorstore.as_retriever(), memory=memory)
        history = st.session_state.get('chat_history', [])
        response, response_time = generate_response(chain, history, query_question)
        history.append({"role": "user", "content": query_question})
        history.append({"role": "assistant", "content": response})
        st.session_state.chat_history = history

        return jsonify({"response": response}), 200

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
