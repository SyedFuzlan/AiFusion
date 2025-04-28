import os
import base64
import gc
import tempfile
import time
import uuid
import logging
import asyncio
import torch
import httpx
import streamlit as st
from IPython.display import Markdown, display
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from streamlit_option_menu import option_menu
from langchain_ollama import ChatOllama, OllamaEmbeddings, OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
import pandas as pd

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Initialize asyncio event loop to avoid RuntimeError
def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

ensure_event_loop()

# Constants
PDF_STORAGE_PATH = 'document_store/pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:7b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:7b")

# Function to calculate attendance percentage
def calculate_attendance_percentage(username, roll_number):
    file_path = r'G:\My Drive\Ntg\Book2.xlsx'
    try:
        attendance_data = pd.read_excel(file_path, sheet_name='Sheet1', header=0)
        attendance_data.columns = attendance_data.columns.str.strip().str.lower().str.replace(" ", "_")
        attendance_data['student_name'] = attendance_data['student_name'].str.strip()
        student_data = attendance_data[(attendance_data['student_name'] == username) & (attendance_data['roll_number'] == roll_number)]
        
        if student_data.empty:
            st.sidebar.error("No data found for the provided username and roll number.")
            return None

        attendance_columns = [col for col in student_data.columns if col not in ['student_name', 'roll_number', 'total_percentage']]
        if len(attendance_columns) == 0:
            st.sidebar.error("No valid attendance columns found. Please verify the data structure in the Excel file.")
            return None

        student_data[attendance_columns] = student_data[attendance_columns].fillna("Absent")
        total_classes = len(attendance_columns)
        classes_attended = student_data[attendance_columns].apply(lambda row: row.eq('Present').sum(), axis=1).values[0]
        attendance_percentage = (classes_attended / total_classes) * 100

        return f"Attendance percentage for {username} ({roll_number}): {attendance_percentage:.2f}%"
    except Exception as e:
        st.sidebar.error(f"An error occurred: {e}")
        return None

# Streamlit session state initialization
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.roll_number = ""

# Login functionality
if not st.session_state.logged_in:
    st.title("AI Fusion")
    st.title("Login")
    st.session_state.username = st.text_input("Enter your name")
    st.session_state.roll_number = st.text_input("Enter your roll number")

    if st.button("Login"):
        if st.session_state.username and st.session_state.roll_number:
            st.session_state.logged_in = True
        else:
            st.error("Please enter both your name and roll number.")
else:
    st.title("🧠 AI Fusion")
    st.caption("🚀 Your AI Pair Programmer and Intelligent Document Assistant")
    st.subheader(f"Hi {st.session_state.username}!")

    # Sidebar configuratio
    def ensure_event_loop():
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    ensure_event_loop()

    # Enable detailed logging for debugging
    logging.basicConfig(level=logging.DEBUG)

    # PyTorch registration check to debug interaction with Streamlit
    try:
        _ = torch.classes
    except RuntimeError as e:
        logging.error(f"PyTorch registration error: {e}")

    # Configure HTTPX client with extended timeout
    custom_timeout = httpx.Timeout(600.0, connect=60.0, read=600.0, write=600.0)
    custom_client = httpx.Client(timeout=custom_timeout)

    # Initialize the Ollama LLM instance
    ollama_instance = Ollama(model="deepseek-r1:7b", httpx_client=custom_client, request_timeout=600.0)

    # Streamlit session state initialization
    if "id" not in st.session_state:
        st.session_state.id = uuid.uuid4()
        st.session_state.file_cache = {}

    session_id = st.session_state.id

    # Cache LLM initialization
    @st.cache_resource
    def load_llm():
        llm = Ollama(model="deepseek-r1:7b", request_timeout=600.0)
        return llm

    def reset_chat():
        st.session_state.messages = []
        st.session_state.context = None
        gc.collect()

    def display_pdf(file):
        st.markdown("### PDF Preview")
        base64_pdf = base64.b64encode(file.read()).decode("utf-8")
        pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                            style="height:100vh; width:100%">
                        </iframe>"""
        st.markdown(pdf_display, unsafe_allow_html=True)
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        selected_model = st.selectbox(
            "Choose Model",
            ["deepseek-r1:1.5b", "deepseek-r1:7b"],
            index=0
        )
        st.divider()
        
        # File Upload Section
        st.header("Add your documents!")
        uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

        if uploaded_file:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    file_key = f"{session_id}-{uploaded_file.name}"
                    st.write("Indexing your document...")

                    if file_key not in st.session_state.get("file_cache", {}):
                        if os.path.exists(temp_dir):
                            loader = SimpleDirectoryReader(
                                input_dir=temp_dir,
                                required_exts=[".pdf"],
                                recursive=True
                            )
                        else:
                            st.error("Could not find the file you uploaded, please check again...")
                            st.stop()
                        
                        docs = loader.load_data()

                        # Initialize LLM and embedding model
                        llm = load_llm()
                        embed_model = HuggingFaceEmbedding(
                            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Optimized embedding model
                            trust_remote_code=True
                        )
                        Settings.embed_model = embed_model
                        index = VectorStoreIndex.from_documents(docs, show_progress=True)

                        # Limit query results and simplify the query engine
                        Settings.llm = llm
                        query_engine = index.as_query_engine(streaming=True, num_results=2)

                        # Define simplified prompt template
                        qa_prompt_tmpl_str = (
                            "Context:\n"
                            "{context_str}\n"
                            "Question: {query_str}\n"
                            "Answer:"
                        )
                        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
                        query_engine.update_prompts(
                            {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                        )
                        
                        st.session_state.file_cache[file_key] = query_engine
                    else:
                        query_engine = st.session_state.file_cache[file_key]

                    st.success("Ready to Chat!")
                    display_pdf(uploaded_file)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.stop()
        st.divider()
        
        # Button to display attendance percentage
        if st.button("Check Attendance Percentage"):
            if 'username' in st.session_state and 'roll_number' in st.session_state:
                attendance_percentage = calculate_attendance_percentage(st.session_state.username, st.session_state.roll_number)
                st.sidebar.write(f"Attendance Percentage: {attendance_percentage}")
            else:
                st.sidebar.write("Error: Please log in to access your attendance percentage.")


        st.divider()
        st.markdown("### Model Capabilities")
        st.markdown("""
        - 🐍 Python Expert
        - 🐞 Debugging Assistant
        - 📝 Code Documentation
        - 💡 Solution Design
        - 📘 Document Analysis
        """)
        st.divider()
        st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

    # Create tabs for Chatbot and RAG functionalities
    selected_tab = option_menu(
        menu_title="Select Functionality",
        options=["Chatbot", "RAG"],
        icons=["chat", "file-earmark-text"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

    # initiate the chat engine
    llm_engine=ChatOllama(
        model=selected_model,
        base_url="http://localhost:11434",
        temperature=0.3
    )

    # System prompt configuration
    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are an expert AI coding assistant. Provide concise, correct solutions "
        "with strategic print statements for debugging. Always respond in English."
    )
    
    # Session state management
    if "message_log" not in st.session_state:
        st.session_state.message_log = [{"role": "ai", "content": f"Hi {st.session_state.username}! How can I help you code today? 💻"}]
    session_id = st.session_state.id
    if "id" not in st.session_state:
        st.session_state.id = uuid.uuid4()
        st.session_state.file_cache = {}
    session_id = st.session_state.id   

    
    if selected_tab == "Chatbot":
        # Chat container
        chat_container = st.container()

        # Display chat messages
        with chat_container:
            for message in st.session_state.message_log:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Chat input and processing
        user_query = st.chat_input("Type your coding question here...")

        def generate_ai_response(prompt_chain):
            processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
            return processing_pipeline.invoke({})

        def build_prompt_chain():
            prompt_sequence = [system_prompt]
            for msg in st.session_state.message_log:
                if msg["role"] == "user":
                    prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
                elif msg["role"] == "ai":
                    prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
            return ChatPromptTemplate.from_messages(prompt_sequence)

        if user_query:
            # Add user message to log
            st.session_state.message_log.append({"role": "user", "content": user_query})

            # Generate AI response
            with st.spinner("🧠 Processing..."):
                prompt_chain = build_prompt_chain()
                ai_response = generate_ai_response(prompt_chain)
    
    # Add AI response to log
            st.session_state.message_log.append({"role": "ai", "content": ai_response})
    
    # Rerun to update chat display
            st.rerun()
    elif selected_tab == "RAG":
        # RAG container
        if "messages" not in st.session_state:
            reset_chat()

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("What's up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Handle assistant responses with retries
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                start_time = time.time()
                try:
                    for attempt in range(3):
                        try:
                            streaming_response = query_engine.query(prompt)
                            for chunk in streaming_response.response_gen:
                                full_response += chunk
                                message_placeholder.markdown(full_response + "▌")
                            break
                        except httpx.ReadTimeout:
                            wait = 2 ** attempt
                            logging.warning(f"Timeout occurred, retrying in {wait} seconds...")
                            time.sleep(wait)
                            if attempt == 2:
                                raise
                except Exception as e:
                    full_response = f"An error occurred while generating the response: {e}"
                    logging.error(full_response)
                    message_placeholder.markdown(full_response)
                finally:
                    end_time = time.time()
                    logging.info(f"Query processing time: {end_time - start_time:.2f} seconds")

                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})