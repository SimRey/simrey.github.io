# --- Imports ---
import os
import re
import psycopg2
import gradio as gr
from dotenv import load_dotenv
from pydantic import BaseModel, Field



# LangChain imports
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import StructuredTool
from langchain.tools.retriever import create_retriever_tool

# --- Load environment ---
load_dotenv()


from langchain.text_splitter import CharacterTextSplitter

def rag_tool():
    """
    Build a retriever tool from online documents (CV, papers, projects, interests)
    """
    urls = [
        "./assets/extra/CV.pdf",  # local PDF
        "https://simrey.github.io/projects.html",  # HTML
        "https://simrey.github.io/interests.html",  # HTML
    ]

    docs = []

    for url in urls:
        try:
            if url.endswith(".pdf"):
                loader = PyPDFLoader(url)
                docs.extend(loader.load())  # keep per-page
            else:
                loader = WebBaseLoader(url)
                docs.extend(loader.load())
        except Exception as e:
            print(f"[RAG] Failed to load {url}: {e}")

    if not docs:
        print("[RAG] No documents loaded!")
        return None

    # Use simpler CharacterTextSplitter (better for structured text like CVs)
    text_splitter = CharacterTextSplitter(
        separator="\n",     # split by line breaks
        chunk_size=5000,    # max size
        chunk_overlap=100,  # keep some context
    )
    split_docs = text_splitter.split_documents(docs)

    # Build vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(split_docs, embeddings)

    # Use MMR retriever to cover different CV sections
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3})

    return create_retriever_tool(
        retriever=retriever,
        name="retriever_tool",
        description="Retrieve information about Simone Reynoso’s academic background, papers, projects, and interests.",
    )



# ============================================================
# 2. Database + Structured Tool
# ============================================================
# DB connection
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is missing!")

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()
cur.execute(
    """
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name TEXT,
    email TEXT,
    notes TEXT
)
"""
)
conn.commit()


def record_user_details(email: str, name: str = "Name not provided", notes: str = "not provided"):
    """
    Insert a user record into the database.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO users (name, email, notes) VALUES (%s, %s, %s)",
            (name, email, notes[:500]),  # truncate notes to 500 chars
        )
    conn.commit()
    return f"✅ User details recorded for {email}."


# Single-string input tool for ConversationalChatAgent
def record_user_details_tool(input_text: str):
    """
    Expects input in the format:
    "name: John Doe; email: johndoe@gmail.com; notes: interested in contact"
    """
    try:
        # Extract email, name, notes using regex
        name_match = re.search(r"name\s*:\s*(.*?)(;|$)", input_text, re.IGNORECASE)
        email_match = re.search(r"email\s*:\s*(.*?)(;|$)", input_text, re.IGNORECASE)
        notes_match = re.search(r"notes\s*:\s*(.*)", input_text, re.IGNORECASE)

        name = name_match.group(1).strip() if name_match else "Name not provided"
        email = email_match.group(1).strip() if email_match else None
        notes = notes_match.group(1).strip() if notes_match else "not provided"

        if not email:
            return "⚠️ Cannot record user: email missing."

        return record_user_details(email=email, name=name, notes=notes)

    except Exception as e:
        return f"⚠️ Error parsing user details: {e}"


record_user_tool = StructuredTool.from_function(
    func=record_user_details_tool,
    name="record_user_details",
    description=(
        "Store user info from a single string. "
        "Format: 'name: John Doe; email: johndoe@gmail.com; notes: short notes'."
    ),
)

# ============================================================
# 3. LLM + Agent
# ============================================================
tools = [tool for tool in [rag_tool(), record_user_tool] if tool]

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
name = "Simone Reynoso Donzelli"

AGENT_SYSTEM_PROMPT = f"""
    You are acting as {name}.
    You have deep knowledge about {name}'s academic background, research interests, and projects.
    You should always refer to {name} using masculine pronouns (he/him/his).

    TOOLS:
    - Use the `retriever_tool` for factual or research-related questions.
    - Use the `record_user_details` tool if the user shares their email, name, or other contact info.
    - When using record_user_details, you MUST provide info in a single string:
    "name: ...; email: ...; notes: ..."

    GUIDELINES:
    - Be professional, engaging, and concise (as if talking to a potential client or future employer).
    - If you cannot answer something, admit it politely.
    - Encourage users to get in touch by email, and record their details if possible.
    - Always pick the most appropriate tool before answering.

"""

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_chain = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    memory=memory,
    agent_kwargs={"system_message": AGENT_SYSTEM_PROMPT},
)

# ============================================================
# 4. Gradio Chat Interface
# ============================================================
def run_agent(input_text, history):
    """
    Run the agent chain with memory synchronized from Gradio's history.
    """
    try:
        if history and len(memory.chat_memory.messages) == 0:
            for user_msg, ai_msg in history:
                if user_msg:
                    memory.chat_memory.add_user_message(user_msg)
                if ai_msg:
                    memory.chat_memory.add_ai_message(ai_msg)
        response = agent_chain.run(input=input_text)
        return response
    except Exception as e:
        return f"⚠️ An error occurred: {e}"


interface = gr.ChatInterface(
    fn=run_agent,
    chatbot=gr.Chatbot(
        height=500,
        label="ReAct Tooling Agent",
        type="messages",
    ),
    textbox=gr.Textbox(
        placeholder="Ask me anything about Simone Reynoso’s profile",
        container=False,
        scale=7,
    ),
    title="ChatBot - Simone Reynoso Donzelli",
    theme="soft",
    type="messages",
)

if __name__ == "__main__":
    interface.launch()