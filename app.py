from uuid import UUID
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
import os

repo_url = "https://github.com/trinity31/rag-pipeline-challenge"

st.set_page_config(
    page_title="Document GPT",
    page_icon="🤖",
)

api_key = ""

with st.sidebar:
    api_key = st.text_input("OpenAI API key", type="password")

    file = st.file_uploader(
        "Upload a txt, pdf or docx file", type=["pdf", "txt", "docx"]
    )

    st.markdown(
        f'<a href="{repo_url}" target="_blank">GitHub Repository</a>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file, api_key):
    file_content = file.read()
    file_dir = f"./.cache/files"  # 파일을 저장할 디렉토리 경로
    if not os.path.exists(file_dir):  # 디렉토리가 존재하지 않으면
        os.makedirs(file_dir)  # 디렉토리 생성
    file_path = os.path.join(file_dir, file.name)
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    else:
        for message in st.session_state["messages"]:
            send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def format_messages_for_prompt(messages):
    formatted_messages = []
    for message in messages:
        role = message["role"]
        text = message["message"]
        formatted_messages.append(f"{role}: {text}")

    res = "\n".join(formatted_messages)
    print(res)
    return res


def format_history(_):
    if "messages" not in st.session_state:
        return ""
    else:
        return format_messages_for_prompt(st.session_state["messages"])


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):  # on_llm_start(1,2,3,4,...,a=1,b=2,...)
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


if api_key == "":
    st.error("Please enter your OpenAI API key")
    st.stop()
else:
    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
        api_key=api_key,
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
     
     Context: {context}

     {history}
     """,
        ),
        ("human", "{question}"),
    ]
)

st.title("Document GPT")

st.markdown(
    """
Welcome!
\n
Use this chatbot to ask questions to AI about your files!
\n
Upoad your file in sidebar.
"""
)


if file:
    if api_key == "":
        st.error("Please enter your OpenAI API key")
        st.stop()
    retriever = embed_file(file, api_key)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")

    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "history": {} | RunnableLambda(format_history),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )

        with st.chat_message("ai"):
            result = chain.invoke(message)

else:
    st.session_state["messages"] = []
