from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain.utilities import WikipediaAPIWrapper
import streamlit as st
import wikipedia

wikipedia = WikipediaAPIWrapper()

# Set favicon
st.set_page_config(page_title="Streamlit App", page_icon="static/res/favicon.png")

st.markdown(
    '''
    <style>
        .center-image {
            display: flex;
            justify-content: center;
        }
    </style>
    <a href="https://pythonpythonme.netlify.app/index.html">
    <div class="center-image">
    <img src="https://pythonpythonme.netlify.app/PythonPythonME.png" alt="Header image">
    </div>
    </a>
    <p></p>
    <p></p>
    <body>
        <header>
            <div>
                <h1>Streamlit Question Answering App</h1>
                <div class="center-image">
                <h1>🦜 🦚</h1>
                </div>
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
    <style>
        .center-image {
            display: flex;
            justify-content: center;
        }
        .follow-me {
            text-align: center;
        }
        .social-icons {
            display: flex;
            justify-content: center;
            list-style: none;
            padding: 0;
        }
        .social-icons li {
            margin: 0 10px;
        }
    </style>
    <body>
        <div class="center-image">
            <h4>Anoop Johny 🤖</h4>
        </div>
        <div class="center-image">
            <h4>Follow Me</h4>
        </div>
        <div class="center-image">
            <ul class="social-icons">
                <li><a href="https://www.linkedin.com/in/anoop-johny-30a746181/"><img src="https://pythonpythonme.netlify.app/static/res/linkedin.png" width="55" height="55" alt="LinkedIn"></a></li>
                <li><a href="https://github.com/flyfir248"><img src="https://pythonpythonme.netlify.app/static/res/github.png" width="55" height="55" alt="GitHub"></a></li>
                <li><a href="https://pythonpythonme.netlify.app/index.html"><img src="https://pythonpythonme.netlify.app/static/res/web.png" width="55" height="55" alt="Website"></a></li>
                <li><a href="https://medium.com/@anoopjohny2000"><img src="https://pythonpythonme.netlify.app/static/res/medium.png" width="55" height="55" alt="Medium"></a></li>
                <li><a href="https://www.kooapp.com/profile/anoop2DEVLJ"><img src="https://www.kooapp.com/_next/static/media/logoKuSolidOutline.1f4fa971.svg" width="55" height="55" alt="The Koo App" width="55" height="55"></a></li>
                <li><a href="https://www.kaggle.com/anoopjohny"><img src="https://cdn3.iconfinder.com/data/icons/logos-and-brands-adobe/512/189_Kaggle-1024.png" alt="The Kaggle App" width="55" height="55"></a></li>
                <li><a href="https://pythonpythonme.onrender.com/"><img src="https://pythonpythonme.netlify.app/static/res/web.png" width="55" height="55" alt="Website"></a></li>
            </ul>
        </div>
        <footer class="footer">
            <div class="container">
                <div class="row">
                    <div class="center-image">
                        <p class="text-muted">© 2023-2024 PythonPythonME.</p>
                        <p>All rights reserved.</p>
                    </div>
                </div>
            </div>
        </footer>
    </body>
    ''',
    unsafe_allow_html=True
)
