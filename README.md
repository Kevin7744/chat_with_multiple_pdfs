# Chat with Multiple PDFs

This application allows users to interactively chat with multiple PDF documents using a conversational AI model. Users can ask questions related to the uploaded PDFs, and the system will provide relevant responses based on the content of the documents.

## Setup

Before running the application, make sure to set up the necessary virtual environment and install the required dependencies. Follow these steps:

1. **Virtual Environments:**
    - Create a virtual environment (venv) to store the necessary dependencies for the project.
    - It is recommended to use a virtual environment manager like `venv`.

2. **Git Ignore:**
    - Add a `.gitignore` file to tell Git to ignore the `.env` and `venv` directories.
    - This helps in maintaining a clean repository without versioning unnecessary files.

3. **Environment Variables:**
    - Create an `.env` file to store API keys and other sensitive information.
    - Make sure to populate the necessary environment variables in the `.env` file.

4. **Install Dependencies:**
    - Run the following command to install the required Python packages:
        ```bash
        pip install streamlit pypdf2 langchain python-dotenv faiss-cpu openai huggingface_hub sentence_transformers
        ```

## Usage

1. **Run the Application:**
    - Execute the following command to run the Streamlit application:
        ```bash
        streamlit run app.py
        ```

2. **Upload PDFs:**
    - In the sidebar, use the file uploader to upload one or more PDF documents.
    - Click the "Process" button to start processing the uploaded PDFs.

3. **Ask Questions:**
    - Enter your questions in the text input box provided.
    - The system will process your question and provide relevant responses based on the content of the uploaded PDFs.

4. **View Responses:**
    - The responses will be displayed in the main interface, showing a conversation between the user and the AI.

## Code Overview

The main functionality of the application is implemented in the `app.py` file. Here's a brief overview of the key components:

- **PDF Processing:**
    - The `get_pdf_text` function extracts text from the uploaded PDFs.
    - `get_text_chunks` breaks the text into smaller chunks for efficient processing.

- **Vector Store:**
    - The `get_vectorstore` function utilizes OpenAI embeddings and FAISS to create a vector store for the processed text chunks.

- **Conversation Chain:**
    - `get_conversation_chain` sets up a conversational retrieval chain using a ChatOpenAI model and a memory buffer.

- **User Interface:**
    - The Streamlit interface allows users to interact with the application.
    - Users can ask questions, upload PDFs, and view responses in a conversational format.

Feel free to explore and enhance the code based on your specific requirements!
