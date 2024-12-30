import streamlit as st
import os

import langchain.text_splitter 
import langchain_community.vectorstores
from langchain_core.documents import Document
from langchain_community.document_loaders import YoutubeLoader

def get_text_splits_for_pdf_files(pdf_files, chunk_size=1000, chunk_overlap=0):
    all_splits=[]
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        pages_documents = loader.load_and_split() 
        text_splitter = langchain.text_splitter.CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        text_splits = text_splitter.split_documents(pages_documents)
        for spl in text_splits:
            all_splits.append(spl)
    return all_splits

def get_retriever_for_pdf_files(all_splits, k_to_retrieve=4, vectorstore='FAISS', distance_strategy='COSINE'): 
    embeddings = langchain_openai.OpenAIEmbeddings()
    if vectorstore=='FAISS':
        db = langchain_community.vectorstores.FAISS.from_documents(all_splits, embeddings, distance_strategy=distance_strategy)
    retriever = db.as_retriever(search_kwargs={'k': k_to_retrieve})
    return retriever

def main():
    st.title("Youtube Summarizer")
    menu_selection = st.sidebar.selectbox("Menu", ("Main", "About")) # Create a sidebar menu
    # Handle menu selections
    if menu_selection == "Main":
        main_usage()
    elif menu_selection == "About":
        show_about_window()

def main_usage():
    st.write(
        "Provide a Youtube video link below to get a summary of its transcript (only works for videos with captions). "
        "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    )

    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    else:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        link=st.text_input("Youtube Link")
        
        if 'https://www.youtube.com/watch' in link:
            import langgraph_summarization
            loader = YoutubeLoader.from_youtube_url(link, add_video_info=False)   
            document=loader.load()
            summary = langgraph_summarization.summarizer_doc(document)
            st.write(summary)

def show_about_window():
    st.header("About")
    st.write("A simple QA application that works with small text documents through the LLM GPT-4o-mini. Find the code here: https://github.com/alescrnjar/YTB_summarizer")

if __name__ == "__main__":
    main()

