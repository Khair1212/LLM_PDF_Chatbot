# LLM_PDF_Chatbot 

LLM PDF Chatbot is powered by Langchain LLM framework which is a basic chatbot application by uploading pdf files. 
## Basic Features:

* Streamlit UI interface where a user can ask and get answer
* accept (.pdf) document format only 
* Used Langchain LLM framework
* API endpoints created using Flask
* Show the response time when generating the output 
* Responses are shown in json format. Ex:
{
“response”: {streaming text}
}

## High Level Architecture 
<img width="587" alt="Screenshot 2024-02-01 at 1 50 40 PM" src="https://github.com/Khair1212/LLM_PDF_Chatbot/assets/41924102/67249a7f-ff7a-40f1-b3dc-37cbc93f0f64">

## Low Level Architecture
<img width="565" alt="Screenshot 2024-02-01 at 1 50 49 PM" src="https://github.com/Khair1212/LLM_PDF_Chatbot/assets/41924102/4c9db24f-ba63-42b1-bdbf-d4d4e45ba407">

## Application Interface
<img width="836" alt="Screenshot 2024-02-01 at 1 54 57 PM" src="https://github.com/Khair1212/LLM_PDF_Chatbot/assets/41924102/36c9d59d-a452-4289-86c1-2c4b44f9dbea">

<img width="848" alt="Screenshot 2024-02-01 at 1 55 04 PM" src="https://github.com/Khair1212/LLM_PDF_Chatbot/assets/41924102/36fde8be-619a-4cca-83ac-4b77b57c4f1d">

## API Endpoints
<img width="841" alt="Screenshot 2024-02-01 at 2 23 10 PM" src="https://github.com/Khair1212/LLM_PDF_Chatbot/assets/41924102/5874861e-ff71-415e-8d6d-b6947c3f420e">

## Future Scope

* Implement Memory Implementation with Session ID
* Implement checking irrelevant questions which are not related to the pdf file 
* Implement Authentication and Authorization system
* Resolve the Backend and Frontend integration issues