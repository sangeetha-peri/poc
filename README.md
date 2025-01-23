### Multi-Document summarizer and chatbot for querying multiple documents using openAI's GPT-4 turbo model:


This code creates a Streamlit-based application with two main functionalities: document summarization and a chatbot interface for querying document contents. Here's a breakdown:

### Key Features:
1. **Environment Setup**:
   - Uses `dotenv` to load environment variables like the OpenAI API key.
   - Retrieves and initializes the OpenAI client for API interactions.

2. **Utility Functions**:
   - **`get_chat_completion`**: Sends messages to OpenAI's GPT model and retrieves responses.
   - **`summarize`**: Splits text into chunks, summarizes each chunk using OpenAI GPT, and merges the summaries.

3. **Streamlit UI**:
   - **Page Configuration**: Sets the title, layout, and displays a company logo.
   - **Tabs**: 
     - **Summarizer Tab**: Allows users to upload PDFs or Word documents for summarization. Extracts text, processes it through the summarization utility, and displays concise summaries for each file.
     - **Chatbot Tab**: 
       - Accepts PDF uploads, processes the text into smaller chunks using the `RecursiveCharacterTextSplitter`.
       - Generates embeddings using OpenAI and stores them in a FAISS vector database.
       - Enables users to query the documents, retrieving relevant chunks and generating responses via OpenAI GPT.

4. **Document Handling**:
   - Supports PDF and Word file uploads for both summarization and chatbot functionalities.
   - Extracts text from uploaded documents for further processing.

5. **Embedding and Search**:
   - Creates text embeddings using OpenAI models.
   - Utilizes FAISS for efficient similarity search in the document chunks.

6. **Question-Answering**:
   - Constructs a prompt template to answer user questions based on extracted document contexts.
   - Uses the LangChain framework to load and execute a QA chain with the GPT model.

7. **Error Handling and User Feedback**:
   - Displays errors or feedback using Streamlit if document processing or API interactions fail.

### Use Cases:
- Quickly summarize long documents for key insights.
- Interactively query document contents to find specific information or answer questions.

### Dependencies:
- Libraries like `PyPDF2` for PDF handling, `docx` for Word documents, LangChain for advanced text processing and retrieval, FAISS for vector search, and OpenAI's API for GPT-based operations.
