#📄 docReader
docReader is a Streamlit application that leverages Generative AI to answer questions based on the content of documents or URLs you provide. It utilizes vector embeddings for both the document and the query, performing a semantic search to deliver precise answers.

✨ Features
Document Parsing: Supports PDF and text files.
Web Content Parsing: Retrieve content from web pages via URLs.
AI-Powered Q&A: Uses GPT-3.5-turbo for generating contextually accurate answers.
Semantic Search: Converts content and questions into vector embeddings to find the most relevant sections.
🚀 Installation
To get started with docReader, follow these steps:

Clone the repository:

bash
Copy code
git clone https://github.com/your-username/docReader.git
cd docReader
Install the required Python packages:

bash
Copy code
pip install streamlit pypdf langchain unstructured unstructured[pdf] tiktoken faiss-cpu langchain-chroma langchain-community
🛠️ Usage
Set up your OpenAI API Key and Organization ID:

Create a .env file in the root of the project and add your OpenAI API key and Organization ID:

makefile
Copy code
OPENAI_API_KEY=your-api-key
ORG_ID=your-org-id
Run the Streamlit app:

bash
Copy code
streamlit run your_app.py
Upload your document or provide a URL:

Upload PDF or text files using the file uploader.
Alternatively, input a URL to retrieve content from a web page.
Ask your question:

Enter a question related to the content you've uploaded or the URL you've provided.
The app will perform a semantic search and generate an accurate answer.
📦 Dependencies
Streamlit: For creating the web application.
OpenAI: For utilizing GPT-3.5-turbo.
LangChain: For handling document loading, embedding, and retrieval.
FAISS: For performing fast similarity search.
PyPDF: For reading PDF files.
Unstructured: For loading and splitting documents.
📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

🤝 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

