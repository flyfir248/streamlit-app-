import streamlit as st # import streamlit for the app

# from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import load_tools  # langchain stuff
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

import os  # importing os

from apikey import apikey  # the apikey to be connected to it

os.environ["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"]=apikey

# intialize the agent
llm =OpenAI(temperature=0)
tools =load_tools(['wikipedia'],llm=llm)
agent=initialize_agent(tools,llm,AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)

# collect user prompt
input =st.text_input('Prompt>>> ')
#wikipedia =WikipediaAPIWrapper()  ## current tool set

if input :
    text = agent.run(input)
    st.text(text)
