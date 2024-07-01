## Chat with PDF

Chat with PDF is an interactive application that allows users to converse with textual content within PDF documents. Using the power of LangChain and Streamlit, this app enables efficient information retrieval by answering questions based on the content of uploaded documents.

## Features
- Upload PDF documents directly into the application.
- Ask natural language questions and receive accurate, context-aware answers.
- Used state-of-the-art language models for response generation.

## Installation

Before running the application, ensure you have Python installed on your system. To install the required dependencies, run the following command:

```sh
pip install streamlit PyMuPDF langchain-openai dotenv
```

## Usage

To start the application, navigate to the app's directory and run:

```sh
streamlit run app.py
```

The application will be available in your web browser at `localhost:8501` by default.

## How it works

1. **Load documents**: Users can upload PDF documents through the web interface.
2. **Extract text & Pre-process**: Text is extracted from the PDF and prepared for processing.
3. **Split documents into chunks**: The application splits documents into smaller, manageable text chunks.
4. **Vector store for Retrieval**: Chunks are indexed in a vector store for efficient retrieval.
5. **LLM for Generation**: Utilizes a large language model (LLM) to generate responses to user queries.
6. **Query & Answer**: The user can query the document content, and the application provides answers.

