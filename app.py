from transformers import AutoTokenizer, pipeline
from langchain.utilities import WikipediaAPIWrapper
import torch
import streamlit as st

st.title("Streamlit Question Answering App 🦜 🦚")

# Load the tokenizer for Falcon LLM
model_name = "tiiuae/falcon-40b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a pipeline for text generation using Falcon LLM
pipeline_falcon = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

# User input
question_input = st.text_input("Question:")

if question_input:
    # Extract keywords from the question input
    keywords = question_input.split()

    # Fetch context information using the Wikipedia toolkit based on keywords
    wikipedia = WikipediaAPIWrapper()
    context_input = wikipedia.run(' '.join(keywords))

    # Generate extended answer using Falcon LLM
    prompt = f"{context_input}\nQuestion: {question_input}\nAnswer:"
    sequences = pipeline_falcon(
        prompt,
        max_length=200,  # Adjust the max length as per your requirement
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Extract the generated text from Falcon LLM output
    generated_text = sequences[0]['generated_text'].replace(prompt, "").strip()

    # Display the answer
    st.text_area("Answer:", generated_text)
