from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain.utilities import WikipediaAPIWrapper
import streamlit as st

st.markdown(
    '''
    <head>
        <link rel="icon" type="image/png" href="{{ url_for('static', filename='res/favicon.png') }}">
    </head>
    <body>
        <header>
            <div>
                <img src="{{ url_for('static', filename='res/icon.png') }}" alt="Main Icon">
                <h1>Streamlit Question Answering App 🦜 🦚</h1>
            </div>
        </header>
    </body>
    ''',
    unsafe_allow_html=True
)

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

st.markdown(
    '''
    <footer class="footer">
            <div class="container">
                <div class="row">
                    <div class="col-md-4">
                        <h4>Anoop Johny 🤖</h4>
                    </div>
                    <div class="col-md-4 text-center">
                        <h4>Follow Me</h4>
                        <ul class="social-icons">
                            <a href="https://www.linkedin.com/in/anoop-johny-30a746181/"><img src="{{ url_for('static', filename='res/linkedin.png') }}" alt="LinkedIn"></a>
                            <a href="https://github.com/flyfir248"><img src="{{ url_for('static', filename='res/github.png') }}" alt="GitHub"></a>
                            <a href="https://pythonpythonme.netlify.app/index.html"><img src="{{ url_for('static', filename='res/web.png') }}" alt="Website"></a>
                            <a href="https://medium.com/@anoopjohny2000"><img src="{{ url_for('static', filename='res/medium.png') }}" alt="Medium"></a>
                            <a href="https://www.kooapp.com/profile/anoop2DEVLJ"><img src="https://www.kooapp.com/_next/static/media/logoKuSolidOutline.1f4fa971.svg" alt="The Koo App" width="55" height="55"></a>
                        </ul>
                    </div>
                </div>
                <div class="row">
                    <div class="col-md-12">
                        <p class="text-muted">© 2023-2024 PythonPythonME.</p>
                        <p>All rights reserved.</p>
                    </div>
                </div>
            </div>
        </footer>
    ''',
    unsafe_allow_html=True
)