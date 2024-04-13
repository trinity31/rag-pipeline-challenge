import streamlit as st
from openai import OpenAI
import json
import requests
from bs4 import BeautifulSoup
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.retrievers import WikipediaRetriever
import time
import asyncio
import httpx


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# def duckduckgo_search(inputs):
#     ddg = DuckDuckGoSearchAPIWrapper()
#     query = inputs["query"]
#     result = ddg.run(query)
#     return result
def duckduckgo_search(inputs):
    # DuckDuckGo의 API 또는 결과 페이지를 스크레이핑하기 위한 URL 설정
    query = inputs["query"]
    url = f"https://html.duckduckgo.com/html/?q={query}"

    # httpx.Client를 사용하여 동기적으로 HTTP 요청 수행
    with httpx.Client() as client:
        response = client.get(url)

    # 검색 결과 처리
    # 이 부분은 실제 필요에 따라 HTML 파싱 또는 API 응답 구조에 맞게 조정해야 합니다.
    print(response.status_code)
    if response.status_code == 200:
        # 예시에서는 응답의 HTML 텍스트를 반환하고 있습니다.
        print(response.text)
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
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "duckduckgo_search",
    #         "description": "Searches DuckDuckGo for the given query",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "query": {
    #                     "type": "string",
    #                     "description": "The query to search for",
    #                 }
    #             },
    #             "required": ["query"],
    #         },
    #     },
    # },
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
    # "duckduckgo_search": duckduckgo_search,
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
        print(f"Calling function: {function.name} with arg {function.arguments}")
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
        st.write(f"{message.role}: {message.content[0].text.value}")


def send_message(thread_id, content):
    client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=content
    )


def save_chat_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_chat_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_chat_message(message, role)


def paint_chat_history():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    else:
        for message in st.session_state["messages"]:
            send_chat_message(message["message"], message["role"], save=False)


repo_url = "https://github.com/trinity31/rag-pipeline-challenge"

api_key = st.session_state.get("api_key", "")
assistant_id = st.session_state.get("assistant_id", "")
assistant_name = st.session_state.get("assistant_name", "")
thread_id = st.session_state.get("thread_id", "")
run_id = st.session_state.get("run_id", "")

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
            Welcome!
            \n
            I'm your Research AI Assistant.
            \n
        """
    )

    if run_id != "":
        run = get_run(run_id, thread_id)
        if run.status == "completed":
            st.write("Research completed!")
            get_messages(thread_id)
        elif run.status == "in_progress":
            st.write("In progress...")
            # get_messages(thread_id)
            time.sleep(3)
            st.rerun()
        elif run.status == "requires_action":
            st.write("Processing action...")
            submit_tool_outputs(run_id, thread_id)
            time.sleep(5)
            st.rerun()

    input = st.text_input(
        "What do you want to research about? ",
        placeholder="e.g) I want to research about Artificial Intelligence",
    )
    if input:
        if thread_id == "":
            thread = client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": input,
                    }
                ]
            )
            thread_id = thread.id
            st.session_state["thread_id"] = thread_id
        else:
            send_message(thread_id, input)

        st.write("Sending message...")

        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )

        run_id = run.id
        st.session_state["run_id"] = run_id

        st.rerun()
