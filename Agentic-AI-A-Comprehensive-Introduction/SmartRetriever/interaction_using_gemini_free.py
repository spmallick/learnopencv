import typer
import os
from typing import Optional,List
from phi.agent import Agent
from phi.knowledge.text import TextKnowledgeBase
from phi.storage.agent.postgres import PgAgentStorage
from phi.vectordb.chroma import ChromaDb
from phi.embedder.google import GeminiEmbedder
from phi.model.google import Gemini


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

api_key = os.getenv('GOOGLE_API_KEY')

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

knowledge_base = TextKnowledgeBase(
    path="/home/opencvuniv/RAG_Agent/results.txt",
    # Table name: ai.text_documents
    vector_db=ChromaDb(
        collection="text_documents",
        path="/home/opencvuniv/RAG_Agent/",
        embedder=GeminiEmbedder(api_key=api_key),
    ),
)

knowledge_base.load()

storage=PgAgentStorage(table_name="rag_agent",db_url=db_url)

def rag_agent(new: bool = False, user: str = "user"):
    session_id: Optional[str] = None

    if not new:
        existing_sessions: List[str] = storage.get_all_session_ids(user)
        if len(existing_sessions) > 0:
            session_id = existing_sessions[0]

    agent = Agent(
        session_id=session_id,
        user_id=user,
        model=Gemini(id="gemini-2.0-flash-exp"),
        knowledge_base=knowledge_base,
        api_key=api_key,
        storage=storage,
        # Show tool calls in the response
        show_tool_calls=True,
        # Enable the assistant to search the knowledge base
        search_knowledge=True,
        # Enable the assistant to read the chat history
        read_chat_history=True,
    )
    if session_id is None:
        session_id = agent.session_id
        print(f"Started Run: {session_id}\n")
    else:
        print(f"Continuing Run: {session_id}\n")

    agent.cli_app(markdown=True)

if __name__=="__main__":
    typer.run(rag_agent)
