import locale
import time
import torch
import streamlit as st
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline

locale.getpreferredencoding = lambda: "UTF-8"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)
if device == 'cuda':
    print(torch.cuda.get_device_name(0))

# Load model and tokenizer
bnb_config = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16,
}

origin_model_path = "tiiuae/falcon-40b-instruct"
token = "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"  # Replace with your Hugging Face token

model = AutoModelForCausalLM.from_pretrained(origin_model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(origin_model_path, use_auth_token=token)

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=300,
    temperature=0.3,
    do_sample=True,
)
mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Function to display text progressively as it is written
def display_text_progressively(text, placeholder):
    current_text = ""
    words = text.split()
    for word in words:
        current_text += word + " "
        placeholder.write(current_text)
        time.sleep(0.1)

# Function to initialize LangChain QA chain
def initialize_qa_chain(file_path):
    loader = CSVLoader(file_path=file_path)
    data = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunked_docs = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    db = FAISS.from_documents(chunked_docs, embeddings)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 4})
    return ConversationalRetrievalChain.from_llm(mistral_llm, retriever, return_source_documents=True)

# Function to display the chatbot interface
def chatbot_interface(qa_chain):
    st.markdown("## Chatbot Interface 🧠")
    st.write("Chatbot is ready! Ask a custom question or select one from the list below.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    standard_questions = ["Who ran the most?", "Which player has the best performance?"]
    selected_question = st.selectbox("Choose a question to ask:", [""] + standard_questions)
    user_input = st.text_input("You (custom question):", key="user_input")

    response_placeholder = st.empty()

    if selected_question or user_input:
        query = selected_question if selected_question else user_input
        try:
            result = qa_chain.invoke({"question": query, "chat_history": st.session_state.chat_history})
            response = result['answer']
            st.session_state.chat_history.append((query, response))
            display_text_progressively(response, response_placeholder)
        except Exception as e:
            st.error(f"Error while retrieving the answer: {e}")

    for query, res in st.session_state.chat_history:
        st.write(f"**You:** {query}")
        st.write(f"{res}")

# Streamlit app
st.set_page_config(page_title="CSV Column Selector & Chatbot", layout="wide")

st.title("Chatbot for CSV Files Interaction")
st.markdown("Drag and drop a CSV file to begin!")

uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])

if uploaded_file:
    st.success("File successfully uploaded!")
    df = pd.read_csv(uploaded_file)

    st.markdown("### Select Columns to Filter")
    selected_columns = st.multiselect(
        "Choose the columns you want to keep:",
        options=df.columns.tolist(),
        default=df.columns.tolist(),
        help="Select the columns you are interested in."
    )

    if selected_columns:
        filtered_df = df[selected_columns]
        st.success("Filtering successful! Here's the filtered data:")
        st.dataframe(filtered_df)

        st.markdown("## 🤖 Chatbot Ready!")
        qa_chain = initialize_qa_chain(uploaded_file)
        chatbot_interface(qa_chain)
    else:
        st.warning("Please select at least one column to proceed!")
else:
    st.info("Please upload a CSV file to begin.")
