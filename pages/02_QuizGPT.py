import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser
import json

repo_url = "https://github.com/trinity31/rag-pipeline-challenge"
api_key = st.session_state.get("api_key", "")

st.set_page_config(
    page_title="Quiz GPT",
    page_icon="❓",
)

st.title("Quiz GPT")

with st.sidebar:
    api_key = st.text_input("OpenAI API key", type="password")
    st.session_state["api_key"] = api_key
    if api_key:
        st.caption("✔️ API key is set.")
    else:
        st.caption("❌ API key is not set.")

    st.markdown(
        f'<a href="{repo_url}" target="_blank">GitHub Repository</a>',
        unsafe_allow_html=True,
    )

##########################
### Langchain code - start
##########################
function_create_quiz = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```json", "").replace("```", "")
        return json.loads(text)


output_parser = JsonOutputParser()

if api_key == "":
    st.error("Please enter your OpenAI API key")
    st.stop()
else:
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-0125",
        streaming=True,
        callbacks=[
            StreamingStdOutCallbackHandler(),
        ],
        api_key=api_key,
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            function_create_quiz,
        ],
    )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.

    Question's difficulty level: {difficulty}
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
""",
        ),
    ]
)

questions_chain = questions_prompt | llm


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, difficulty):
    # _docs: not hashing, topic: hashing
    response = questions_chain.invoke(
        {
            "context": format_docs(_docs),
            "difficulty": difficulty,
        }
    )
    # st.write(response)
    result = response.additional_kwargs["function_call"]["arguments"]
    res = json.loads(result)
    # st.write(res)
    return res


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5, lang="en")
    docs = retriever.get_relevant_documents(term)
    return docs


##########################
### Langchain code - end
##########################

##########################
### Streamlit code - start
##########################

with st.sidebar:
    docs = None
    topic = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    difficulty = st.selectbox(
        "Choose the difficulty level.",
        (
            "Easy",
            "Difficult",
        ),
    )

    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt or .pdf file", type=["docx", "txt", "pdf"]
        )
        if file:
            if api_key == "":
                st.error("Please enter your OpenAI API key")
                st.stop()
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)


if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    # start = st.button("Generate Quiz")
    # st.write(docs)
    # if start:
    try:
        response = run_quiz_chain(
            docs,
            topic if topic else file.name,
            difficulty,
        )
        # st.write(response)
        correct_answers = 0
        total_questions = len(response["questions"])
        with st.form("questions_form"):
            for question in response["questions"]:
                st.write(question["question"])
                value = st.radio(
                    "Select an option",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                    key=question["question"],
                )
                if {"answer": value, "correct": True} in question["answers"]:
                    correct_answers += 1
                    st.success("Correct!")
                elif value is not None:
                    st.error("Wrong")
                    # 만점이 아닌 경우에만 제출 버튼을 표시

            submitted = st.form_submit_button(
                "Submit", disabled=(correct_answers == total_questions)
            )

        if correct_answers < total_questions:
            st.write(f"You got {correct_answers} out of {total_questions} correct.")
        else:
            st.success(f"Congratulations! You got all {total_questions} correct!")
            st.balloons()
    except NameError:
        st.error("Please upload a file or search Wikipedia to generate a quiz.")
    # st.write(response)

##########################
### Streamlit code - end
##########################
