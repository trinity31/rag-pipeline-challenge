import streamlit as st
from openai import OpenAI
import json
import requests
from bs4 import BeautifulSoup
from langchain.retrievers import WikipediaRetriever
import time
import httpx


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def duckduckgo_search(inputs):
    query = inputs["query"]
    api_key = st.secrets["SERPAPI_KEY"]
    url = f"https://serpapi.com/search?engine=duckduckgo&q={query}&api_key={api_key}"

    # httpx.Client를 사용하여 동기적으로 HTTP 요청 수행
    with httpx.Client() as client:
        response = client.get(url)

    if response.status_code == 200:
        return response.text
    else:
        return "Search failed with status: " + str(response.status_code)


def wikipedia_search(inputs):
    query = inputs["query"]
    retriever = WikipediaRetriever(top_k_results=5, lang="en")
    docs = retriever.get_relevant_documents(query)
    return format_docs(docs)


def website_scrape(inputs):
    url = inputs["url"]
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    # 모든 <p> 태그의 텍스트를 추출하고 결합
    content = " ".join([p.text for p in soup.find_all("p")])
    return content


def save_to_file(inputs):
    data = inputs["data"]
    with open(f"search_result.txt", "w") as f:
        f.write(data)
    return f"Information saved in result.txt file."


functions = [
    {
        "type": "function",
        "function": {
            "name": "duckduckgo_search",
            "description": "Searches DuckDuckGo for the given query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Searches Wikipedia for the given query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "website_scrape",
            "description": "Scrapes the content of the given website",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the website to scrape",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_file",
            "description": "Saves the given data to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string",
                        "description": "The data to save",
                    }
                },
                "required": ["data"],
            },
        },
    },
]

functions_map = {
    "duckduckgo_search": duckduckgo_search,
    "wikipedia_search": wikipedia_search,
    "website_scrape": website_scrape,
    "save_to_file": save_to_file,
}


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        function = action.function
        st.write(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action.id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs,
    )


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    for message in messages:
        with st.chat_message(message.role):
            st.markdown(message.content[0].text.value)


def send_message(thread_id, content):
    client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=content
    )


def send_chat_message(message, role):
    with st.chat_message(role):
        st.markdown(message)


repo_url = "https://github.com/trinity31/rag-pipeline-challenge"

api_key = st.session_state.get("api_key", "")
assistant_id = st.session_state.get("assistant_id", "")
assistant_name = st.session_state.get("assistant_name", "")
thread_id = st.session_state.get("thread_id", "")
run_id = st.session_state.get("run_id", "")

if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False

st.set_page_config(
    page_title="Research GPT",
    page_icon="🔎",
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
    client = OpenAI(api_key=api_key)

    # Create an assistant if it doesn't exist
    if assistant_id == "":
        # List all the assistants
        my_assistants = client.beta.assistants.list(
            order="desc",
            limit="10",
        )

        # Find the assistant named "Research AI Assistant"
        for assistant in my_assistants.data:
            if assistant.name == "Research AI Assistant":
                print(
                    f"Found already existing assistant: {assistant.name}, {assistant.id}"
                )
                assistant_id = assistant.id
                assistant_name = assistant.name
                st.session_state["assistant_id"] = assistant_id
                st.session_state["assistant_name"] = assistant_name
                break

        # Create a new assistant if it doesn't exist
        if assistant_id == "":
            assistant = client.beta.assistants.create(
                name="Research AI Assistant",
                instructions="""
                        You are a research specialist.
                        Take a deep breath and proceed with the following step.
                        1. Search for the information about a query using DuckDuckGo.
                        2. Search for the information about a query using Wikipedia.
                        3. Extract the content of a website if any url is included in the search result.
                        4. Save all the search results in a text file, and it should contain all the information you have found in step 1, 2, and 3.
                        """,
                model="gpt-4-turbo-preview",
                tools=functions,
            )
            st.session_state["assistant_id"] = assistant.id
            st.session_state["assistant_name"] = assistant.name
            assistant_id = assistant.id
            assistant_name = assistant.name

            print(f"Created a new assistant with id: {assistant_id}")

    st.title(assistant_name)
    st.markdown(
        """
            Welcome! I'm your Research AI Assistant.
            \n
            I can help you with your research.
            \n
            Please enter your query below to get started.
            \n
        """
    )

    keyword = st.session_state["keyword"] if "keyword" in st.session_state else ""

    if run_id != "":
        run = get_run(run_id, thread_id)
        if run.status == "completed":
            st.success("Research completed!")
            get_messages(thread_id)
            with open("search_result.txt", "rb") as file:
                btn = st.download_button(
                    label="Download result",
                    data=file,
                    file_name=f"search_{keyword}.txt",
                    mime="text/plain",
                    on_click=lambda: setattr(st.session_state, "button_clicked", True),
                )
        elif run.status == "in_progress":
            with st.status("In progress..."):
                st.write("Waiting for the AI to respond...")
                time.sleep(3)
                st.rerun()
        elif run.status == "requires_action":
            with st.status("Processing action..."):
                submit_tool_outputs(run_id, thread_id)
                time.sleep(5)
                st.rerun()

    if "reset_input" in st.session_state and st.session_state["reset_input"]:
        st.session_state["input"] = ""
        st.session_state["reset_input"] = False

    input_value = st.text_input(
        "What do you want to research about? ",
        value=st.session_state.get("input", ""),
        key="input",
        placeholder="e.g) I want to research about Artificial Intelligence",
    )

    if input_value:
        if thread_id == "":
            thread = client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": input_value,
                    }
                ]
            )
            thread_id = thread.id
            st.session_state["thread_id"] = thread_id
        else:
            send_message(thread_id, input_value)

        # st.write("Sending message...")

        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )

        run_id = run.id
        st.session_state["run_id"] = run_id
        st.session_state["reset_input"] = True
        st.session_state["keyword"] = input_value

        st.rerun()
