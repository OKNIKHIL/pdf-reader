import streamlit as st
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
headers={
    "authorization":st.secrets[OPENAI_API_KEY]
}

# Sidebar contents
with st.sidebar:
    st.title('🤗💬  Chat App')
    st.markdown('''
    <style>
        /* Add your custom CSS code here */
        .sidebar-content {
            background-color: #f0f0f0;
            padding: 20px;
            border-radius: 10px;
        }
        .sidebar-title {
            color: #333333;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
    </style>

    <div class="sidebar-content">
        <div class="sidebar-title">About</div>
        <p>This app is an LLM-powered chatbot built using:</p>
        <ul>
            <li><a href="https://streamlit.io/">Streamlit</a></li>
            <li><a href="https://python.langchain.com/">LangChain</a></li>
            <li><a href="https://platform.openai.com/docs/models">OpenAI LLM model</a></li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)


def main():
    st.header("Chat with PDF 💬")
    chat_history = set()  # Initialize an empty set for chat history
    question_counter = 0  # Initialize a counter for questions

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            import openai
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        query = st.text_input("Ask questions about your PDF file:")
        
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
     
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

            chat_history.add(query)
            question_counter += 1

        if chat_history:
            st.sidebar.subheader("Chat History")
            for question in chat_history:
                st.sidebar.write(f"Q: {question}")

if __name__== '__main__':
    main()
