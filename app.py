from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain.utilities import WikipediaAPIWrapper
import streamlit as st
import base64
from PIL import Image

image = Image.open('res/icon.png')
resized_image = image.resize((330, 300))  # Set the desired width and height here

# Convert image to bytes and encode as base64
image_bytes = resized_image.tobytes()
image_base64 = base64.b64encode(image_bytes).decode()

# Center the image
centered_image = f'<div style="display: flex; justify-content: center;"><a href="https://pythonpythonme.netlify.app/index.html"><img src="data:image/png;base64,{image_base64}" alt="Image" width="200" height="200"></a></div>'

st.markdown(centered_image, unsafe_allow_html=True)

st.title("Streamlit Question Answering App 🦜 🦚")

# Load the question answering model and tokenizer
model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a pipeline for question answering
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

# User input
question_input = st.text_input("Question:")

if question_input:
    # Extract keywords from the question input
    keywords = question_input.split()

    # Fetch context information using the Wikipedia toolkit based on keywords
    wikipedia = WikipediaAPIWrapper()
    context_input = wikipedia.run(' '.join(keywords))

    # Prepare the question and context for question answering
    QA_input = {
        'question': question_input,
        'context': context_input
    }

    # Get the answer using the question answering pipeline
    res = nlp(QA_input)

    # Display the answer
    st.text_area("Answer:", res['answer'])
    st.write("Score:", res['score'])