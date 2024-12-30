import streamlit as st
import os

from langchain_community.document_loaders import YoutubeLoader

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

