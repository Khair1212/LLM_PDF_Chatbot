import streamlit as st
from dotenv import load_dotenv
import uuid
import requests

def generate_session_id():
    return str(uuid.uuid4())


# Main Function
def main():
    st.header('PDF CHATBOT ')
    load_dotenv()

    session_id = generate_session_id()

    # upload a pdf file and extract text (200mb limit)
    pdf = st.file_uploader("Upload you PDF", type="pdf")
    # query = st.text_input("Enter a question:",  "", key="query_input")

    if pdf is not None:
        # print("Session ID:", session_id)
        # Save the uploaded file temporarily
        with open(f"/tmp/{pdf.name[:-4]}.pdf", "wb") as f:
            f.write(pdf.read())

        query = st.text_input("Enter a question:", "")
        # Interaction with the user
        if query:
            # print("Query:", query)
            # print("PDF Name:", pdf.name)

            response = requests.post(f"http://localhost:5000/get_chatbot_response/{session_id}{pdf.name[:-4]}{query}", json={"query": query}, headers={"Content-Type": "application/json"})
            print(response.text)
            response_json = response.json()
            st.write(f"Chatbot: {response_json['response']}")

if __name__ == '__main__':
    main()