import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import unicodedata
import re
import random
import string
import tempfile
from dotenv import load_dotenv

# Set environment variables
load_dotenv()

# Set Tesseract path (adjust for your system)
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
poppler_path = r"C:/poppler-24.08.0/Library/bin"

# Initialize OpenAI model for embeddings and chat
llm = ChatOpenAI(
    model="gpt-4.1-2025-04-14",
    temperature=0,
    max_retries=2,
)

embed_model = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embedding=embed_model)

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Streamlit UI
st.title("Chat With your PDF")
st.write("Upload a PDF and start chatting!")

pdf_file = st.file_uploader("Upload your PDF", type="pdf")


if pdf_file:
    if 'pdf_processed' not in st.session_state:
        # Display processing
        st.write("Getting ready...")

        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_pdf_file:
            tmp_pdf_file.write(pdf_file.read())
            tmp_pdf_path = tmp_pdf_file.name

        # Convert PDF to images
        images = convert_from_path(tmp_pdf_path, poppler_path=poppler_path)

        #  Extract and clean text
        def clean_text(text):
            text = unicodedata.normalize("NFC", text)
            text = re.sub(r"(পৃষ্ঠা|Page)\s*\d+", "", text)
            text = re.sub(r"\n+", "\n", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        docs = []
        for i, img in enumerate(images):
            raw_text = pytesseract.image_to_string(img, lang="ben")
            cleaned = clean_text(raw_text)
            docs.append(Document(page_content=cleaned, metadata={"page": i + 1}))

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=30,
            add_start_index=True
        )

        all_splits = text_splitter.split_documents(docs)

        # Store embeddings in vector store
        vector_store.add_documents(documents=all_splits)

        # Mark PDF as processed to prevent re-processing in future chats
        st.session_state.pdf_processed = True
        st.session_state.vector_store = vector_store

        # Generate a random session ID
        session_id = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        st.session_state['session_id'] = session_id

    else:
        st.write("PDF is ready, you can Chat NOw.")

    # User input field
    prompt = st.chat_input("Ask a question")
    
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve relevant documents based on user input
        retrieved_docs = st.session_state.vector_store.similarity_search(query=prompt, k=4)

        
        docs_contents = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt_with_context = f"Context:\n{docs_contents}\nQuestion: {prompt}\nAnswer:"

        # Generate response from the model
        with st.chat_message("assistant"):
            response = llm.invoke(prompt_with_context)
            st.markdown(response)

        # Add response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response.content})

# View chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
