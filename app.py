from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import streamlit as st

st.title("Streamlit Question Answering App 🦜 🦚")

# Load the question answering model and tokenizer
model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a pipeline for question answering
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

# User input
question_input = st.text_input("Question:")
context_input = st.text_area("Context:")

if question_input and context_input:
    # Prepare the question and context for question answering
    QA_input = {
        'question': question_input,
        'context': context_input
    }

    # Get the answer using the question answering pipeline
    res = nlp(QA_input)

    # Display the answer
    st.write("Answer:", res['answer'])
    st.write("Score:", res['score'])
