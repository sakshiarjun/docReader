#pip install pypdf
#pip install langchain
#pip install -U langchain-community
#pip install unstructured unstructured[pdf]
#pip install tiktoken
#pip install faiss-cpu

import streamlit as st
import pandas as pd
import openai
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredFileLoader
#from langchain_community.document_loaders import OnlinePDFLoader use for remote pdfs
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
#from io import StringIO

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.org_id = os.getenv('ORG_ID')
st.title("docReader")

client = OpenAI()
def get_embedding(text, model='text-embedding-ada-002'):
    response = client.embeddings.create(input=[text], model=model)
    embeddings = response['data'][0]['embedding']
    return embeddings

        
st.write("file uploaded from backend")
loader = UnstructuredFileLoader("/home/sakshi/Desktop/docReader/Taj_Mahal.pdf")  #need filepath, cant get from st.file_uploader
pages = loader.load_and_split()
type(pages[0])
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(pages)
#st.write(text_chunks)

faiss_index = FAISS.from_documents(text_chunks, OpenAIEmbeddings(model='text-embedding-ada-002'))
input_q = st.text_area("Enter question: ")
answers = faiss_index.similarity_search(input_q,k=3)
for i in answers:
    st.write(i.page_content)
    st.markdown("---")
