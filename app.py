import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import time, uuid


def process_pdf(pdf):
    # Extract text from PDF using

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

def generate_session_id():
    return str(uuid.uuid4())

def generate_response(chain, history, query):
    start_time = time.time()  # Record the start time
    result = chain(
        {"question": query, 'chat_history': history}, return_only_outputs=True)
    end_time = time.time()  # Record the end time
    response_time = end_time - start_time
    return result["answer"], response_time


# Main Function
def main():
    st.header('PDF CHATBOT ')
    load_dotenv()

    session_id = generate_session_id()

    # upload a pdf file and extract text (200mb limit)
    pdf = st.file_uploader("Upload you PDF", type="pdf")
    query = st.text_input("Enter a question:", "")

    if pdf is not None:
        text = process_pdf(pdf)
        chunks = text_to_chunks(text)
        vectorstore = get_embeddings(chunks)
        # st.write(chunks)

        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.3),
                                                      retriever=vectorstore.as_retriever(), memory=memory)

        # Interaction with the user
        if query:
            # Fetch the chat history from Streamlit's session state
            history = st.session_state.get('chat_history', [])
            # Generate a response based on the user's question and the chat history
            response, response_time = generate_response(chain, history, query)

            # Display the question and response
            st.write(f"User: {query}")
            st.write(f"ChatBot: {response}")
            st.write(f"Response Time: {response_time}")
            # Update the chat history in the session state
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})
            st.session_state.chat_history = history

if __name__ == '__main__':
    main()

