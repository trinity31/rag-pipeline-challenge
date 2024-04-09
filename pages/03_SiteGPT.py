import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser
import json
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

repo_url = "https://github.com/trinity31/rag-pipeline-challenge"
site_url = "https://developers.cloudflare.com/sitemap.xml"

api_key = st.session_state.get("api_key", "")

st.set_page_config(
    page_title="Cloudflare GPT",
    page_icon="❓",
)

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

if api_key == "":
    st.error("Please enter your OpenAI API key")
    st.stop()
else:
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-0125",
        api_key=api_key,
    )

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                    
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(input):
    answers = input["answers"]
    question = input["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource: {answer['source']}\nDate: {answer['date']}\n"
        for answer in answers
    )

    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", "")
        .replace("\xa0", "")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=200,
    )

    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/workers-ai\/).*",
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 5
    docs = loader.load_and_split(text_splitter=splitter)
    st.write(docs)
    vector_store = FAISS.from_documents(
        docs,
        OpenAIEmbeddings(
            api_key=api_key,
        ),
    )
    return vector_store.as_retriever()


st.markdown(
    """
    # Cloudflare Site GPT
            
    Ask questions about the following content of a Cloudflare Developers Website 

    AI Gateway
    Cloudflare Vectorize
    Workers AI

    https://developers.cloudflare.com.
            
"""
)

if site_url:
    if ".xml" not in site_url:
        with st.sidebar:
            st.error("Wrong URL")
    elif api_key == "":
        with st.sidebar:
            st.error("Please enter your OpenAI API key")
    else:
        retriever = load_website(site_url)
        query = st.text_input(
            "Ask a question about the Cloudflare AI Gateway, Cloudflare Vectorize or Workers AI"
        )
        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )

            result = chain.invoke(query)
            st.write(result.content)
