import streamlit as st
from langchain.utilities import WikipediaAPIWrapper




input =st.text_input('Prompt>>> ')
wikipedia =WikipediaAPIWrapper()  ## current tool set

if input :
    text = wikipedia.run(input)
    st.text(text)
