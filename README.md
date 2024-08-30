
## ✨ Features

- **Document Parsing**: Supports PDF and text files.
- **Web Content Parsing**: Retrieve content from web pages via URLs.
- **AI-Powered Q&A**: Uses GPT-3.5-turbo for generating contextually accurate answers.
- **Semantic Search**: Converts content and questions into vector embeddings to find the most relevant sections.

## 🚀 Installation

To get started with **docReader**, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/docReader.git
   cd docReader
   ```

2. **Install the required Python packages**:
   ```bash
   pip install streamlit pypdf langchain unstructured unstructured[pdf] tiktoken faiss-cpu langchain-chroma langchain-community
   ```

## 🛠️ Usage

1. **Set up your OpenAI API Key and Organization ID**:

   Create a `.env` file in the root of the project and add your OpenAI API key and Organization ID:
   ```
   OPENAI_API_KEY=your-api-key
   ORG_ID=your-org-id
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run your_app.py
   ```

3. **Upload your document or provide a URL**:
   - Upload PDF or text files using the file uploader.
   - Alternatively, input a URL to retrieve content from a web page.

4. **Ask your question**:
   - Enter a question related to the content you've uploaded or the URL you've provided.
   - The app will perform a semantic search and generate an accurate answer.

## 📦 Dependencies

- **Streamlit**: For creating the web application.
- **OpenAI**: For utilizing GPT-3.5-turbo.
- **LangChain**: For handling document loading, embedding, and retrieval.
- **FAISS**: For performing fast similarity search.
- **PyPDF**: For reading PDF files.
- **Unstructured**: For loading and splitting documents.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📷 Images

<img width="451" alt="image" src="https://github.com/user-attachments/assets/27739caf-c527-4bd6-a8c9-0c7d2574556b">






