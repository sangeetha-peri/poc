import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
import tiktoken
from tqdm import tqdm
from typing import List

# Load environment variables
load_dotenv()

# Set up OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Utility Functions for Summarization ---
def get_chat_completion(messages, model='gpt-4-turbo'):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

def summarize(text: str, detail: float = 0.5, model: str = 'gpt-4-turbo') -> str:
    chunks = text.split(".")
    summaries = []
    for chunk in tqdm(chunks, desc="Summarizing chunks"):
        if chunk.strip():  # Avoid processing empty chunks
            messages = [
                {"role": "system", "content": "Rewrite this text in summarized form."},
                {"role": "user", "content": chunk},
            ]
            summaries.append(get_chat_completion(messages, model=model))
    return " ".join(summaries)

# --- Streamlit UI ---
# Set page configuration at the very beginning
st.set_page_config(page_title="Chat PDF", layout="wide")

# Display the logo
st.image("images/company_logo.png", width=200)

st.title("Chat with any documents and summarize PDF files using OpenAI GPT")
st.write("Upload PDFs to summarize their content or chat with them!")

tab1, tab2 = st.tabs(["Summarizer", "Chatbot"])

with tab1:
    uploaded_files = st.file_uploader("Upload PDF or Word files for summarization", type=["pdf", "docx"], accept_multiple_files=True)
    if uploaded_files:
        summaries = []
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_text = ""
            
            if uploaded_file.type == "application/pdf":
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    file_text += page.extract_text()
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = Document(uploaded_file)
                for paragraph in doc.paragraphs:
                    file_text += paragraph.text

            # Summarize each file
            st.write(f"Summarizing: {file_name}")
            summary = summarize(file_text, detail=0.5)
            summaries.append((file_name, summary))

        st.subheader("Summaries:")
        for file_name, summary in summaries:
            st.write(f"### {file_name}")
            st.write(summary)

with tab2:
    with st.sidebar:
        st.title("Chatbot Setup")
        pdf_docs = st.file_uploader("Upload PDF files for chatbot", accept_multiple_files=True)
        if st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                raw_text = ""
                for pdf in pdf_docs:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        raw_text += page.extract_text()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
                text_chunks = text_splitter.split_text(raw_text)
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
                vector_store.save_local("faiss_index")
                st.success("PDFs processed successfully!")

    user_question = st.text_input("Ask a question about the uploaded PDFs:")
    if user_question:
        st.write("Chatbot's reply:")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        model = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY, temperature=0.3)
        prompt_template = """
        Answer the question concisely using the provided context. If the answer is not in the context, say, "The answer is not available in the context."

        Context:\n{context}\n
        Question:\n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write(response["output_text"])
