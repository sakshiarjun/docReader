#pip install pypdf
#pip install langchain
#pip install -U langchain-community
#pip install unstructured unstructured[pdf]
#pip install tiktoken
#pip install faiss-cpu
#pip install langchain-chroma

import streamlit as st
import openai
import os
from openai import OpenAI
#from dotenv import load_dotenv
from os import getenv
#from langchain.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings    
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
openai.api_key = getenv("OPENAI_API_KEY")
openai.org_id = getenv('ORG_ID')
st.title("docReader")

client = OpenAI()
def get_document_text(uploaded_files,file_names):
    docs=[]
    for uploaded_file, file_name in zip(uploaded_files,file_names):
        if file_name.endswith('pdf'):
            pdf_reader = PdfReader(uploaded_file)
            doc_text = ""
            for i, page in enumerate(pdf_reader.pages):
                page = page.extract_text()
                doc = Document(page_content = page, metadata={'page':(i+1)})
                docs.append(doc)
        elif file_name.endswith('txt'):
            txt_reader = uploaded_file.read().decode()
            doc = Document(page_content = txt_reader, metadata={'file_name':file_name})
            docs.append(doc)
    return docs

def get_text_chunks(uploaded_files):
    file_names = [file.name for file in uploaded_files]
    docs = get_document_text(uploaded_files, file_names)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(docs)
    return text_chunks

def generate_response(input_q, retriever):    
    prompt_template = """You are a helpful bot who answers questions based on the content provided below. Answer to the point and in context to the content provided. If question is irrelevant to the content say that 'the question is not relevant to the content given' and don't try to make up an answer.
    content is: {content}
    question is: {input_q}
    give your answer: """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model_name = "gpt-3.5-turbo")
    rag_chain = ({"content": retriever | format_answer, "input_q": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    answer = rag_chain.invoke(input_q)
    return answer

def format_answer(results):
    return "\n".join([result.page_content for result in results])

uploaded_files = st.file_uploader("", accept_multiple_files=True) #size limited to 200MB
st.write("or")
url = st.text_input("Paste page URL")
if uploaded_files:
    text_chunks = get_text_chunks(uploaded_files)
    db = FAISS.from_documents(text_chunks, OpenAIEmbeddings())
    input_q = st.text_area("Question: ")
    if input_q:
        results = db.similarity_search(input_q,k=2)
        retriever = db.as_retriever()
        answer = generate_response(input_q, retriever)
        st.write(answer)
elif url:
    loader = WebBaseLoader(url)

    try:
        data = loader.load_and_split()
    except Exception as e:
        st.error(f"Error loading data from URL: {e}")
        st.stop()
    if not data:
        st.error("No data was retrieved from the provided URL.")
        st.stop()
    try:
        db = FAISS.from_documents(data, OpenAIEmbeddings())
    except Exception as e:
        st.error(f"Error initializing FAISS index: {e}")
        st.stop()
    input_q = st.text_area("Question: ")
    if input_q:
        results = db.similarity_search(input_q, k=2)
        retriever = db.as_retriever()
        answer = generate_response(input_q, retriever)
        st.write(answer)
else:
    st.write(" ")
