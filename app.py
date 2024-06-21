#pip install pypdf
#pip install langchain
#pip install -U langchain-community
#pip install unstructured unstructured[pdf]

import streamlit as st
import pandas as pd
import openai
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
#from io import StringIO

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.org_id = os.getenv('ORG_ID')

st.title("docReader")
#uploaded_file = st.file_uploader("Choose a file (preferably .txt)")

#df = pd.read_fwf(uploaded_file)

#if uploaded_file is not None:
    #stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #st.write(stringio)
    #string_data = stringio.read()
    #st.write(string_data)
        
st.write("file uploaded from backend")
loader = UnstructuredFileLoader("/home/sakshi/Desktop/docReader/Brown-Giving-PsychSci-2003.pdf")  #need filepath, cant get from st.file_uploader
pages = loader.load_and_split()
type(pages[0])
text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(pages)
st.write(text_chunks)

client = OpenAI()
def get_embedding(text, model='text-embedding-ada-002'):
    response = client.embeddings.create(input=[text], model=model)
    embeddings = response['data'][0]['embedding']
    return embeddings
def embed_text_chunks(text_chunks, model='text-embedding-ada-002', output_file='/home/sakshi/Desktop/docReader/embeddings.json'):
    embeddings_dict = {}
    for i, chunk in enumerate(text_chunks):
        embeddings = get_embeddings(chunk, model=model)
        embeddings_dict[f'chunk_{i}'] = embeddings
    with open(output_file, 'w') as f:
        json.dump(embeddings_dict, f)
def print_embeddings_file(file_path='/home/sakshi/Desktop/docReader/embeddings.json'):
    with open(file_path, 'r') as f:
        embeddings_dict = json.load(f)
    st.write(json.dumps(embeddings_dict, indent=4))
print_embeddings_file()

#def response(question, get_embedding()):
    #q_embedding = get_embedding(question)
input_q = st.text_area("Enter question: ")
#output_a = 
