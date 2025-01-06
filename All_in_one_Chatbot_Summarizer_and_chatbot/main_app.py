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
from typing import List, Tuple, Optional

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


def tokenize(text: str) -> List[str]:
    encoding = tiktoken.encoding_for_model('gpt-4-turbo')
    return encoding.encode(text)


def summarize(text: str, detail: float = 0.5, model: str = 'gpt-4-turbo') -> str:
    max_tokens = 4000
    chunks = text.split(".")
    summaries = []
    for chunk in chunks:
        messages = [
            {"role": "system", "content": "Rewrite this text in summarized form."},
            {"role": "user", "content": chunk},
        ]
        summaries.append(get_chat_completion(messages, model=model))
    return " ".join(summaries)

# --- Utility Functions for Chatbot ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question concisely using the provided context. If the answer is not in the context, say, "The answer is not available in the context."

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def chatbot_response(user_question):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# --- Streamlit UI ---
st.title("PDF Summarizer & Chatbot for querying PDF using OpenAI GPT💁")
st.write("Upload PDFs to summarize their content or chat with them!")

tab1, tab2 = st.tabs(["Summarizer", "Chatbot"])

with tab1:
    uploaded_files = st.file_uploader("Upload PDF or Word files for summarization", type=["pdf", "docx"], accept_multiple_files=True)
    if uploaded_files:
        all_text = ""
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    all_text += page.extract_text()
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = Document(uploaded_file)
                for paragraph in doc.paragraphs:
                    all_text += paragraph.text

        st.write("Summarizing the uploaded files...")
        summary = summarize(all_text, detail=0.5)
        st.subheader("Summary:")
        st.write(summary)

with tab2:
    with st.sidebar:
        st.title("Chatbot Setup")
        pdf_docs = st.file_uploader("Upload PDF files for chatbot", accept_multiple_files=True)
        if st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDFs processed successfully!")

    user_question = st.text_input("Ask a question about the uploaded PDFs:")
    if user_question:
        st.write("Chatbot's reply:")
        st.write(chatbot_response(user_question))
