import streamlit as st
from langchain.utilities import WikipediaAPIWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# Load the WikipediaAPIWrapper
wikipedia = WikipediaAPIWrapper()

# Load the Language Model
model_name = "gpt2"  # Choose a language model (e.g., "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", etc.)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


st.title("Streamlit Langchain App : 🦜")
input = st.text_input('Prompt>>> ')

if input:
    # Fetch the Wikipedia data
    wikipedia_data = wikipedia.run(input)

    # Generate text based on the user prompt and Wikipedia data using the Language Model
    prompt = f"{input}\nWikipedia Data: {wikipedia_data}\nGenerated Text:"
    generated_text = text_generator(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']

    # Display the generated text
    st.text_area(generated_text)
