import streamlit as st
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
import os

from apikey import apikey

os.environ["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"] = apikey

llm = OpenAI(temperature=0)
tools = load_tools(['wikipedia'], llm=llm)
agent = initialize_agent(tools, llm, AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

input_prompt = st.text_input('Prompt>>> ')

if input_prompt:
    text = agent.run(input_prompt)
    st.text(text)
