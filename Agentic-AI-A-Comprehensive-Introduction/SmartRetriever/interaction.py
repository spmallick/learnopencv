import typer
import os
from typing import Optional,List
from phi.assistant import Assistant
from phi.knowledge.text import TextKnowledgeBase
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.vectordb.chroma import ChromaDb


from dotenv import load_dotenv
load_dotenv()


db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

knowledge_base = TextKnowledgeBase(
    path="/home/opencvuniv/RAG_Agent/results.txt",
    # Table name: ai.text_documents
    vector_db=ChromaDb(
        collection="text_documents",
        path="/home/opencvuniv/RAG_Agent/",
    ),
)

knowledge_base.load()

storage=PgAssistantStorage(table_name="rag_agent",db_url=db_url)

def rag_agent(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None

    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]

    assistant = Assistant(
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        # Show tool calls in the response
        show_tool_calls=True,
        # Enable the assistant to search the knowledge base
        search_knowledge=True,
        # Enable the assistant to read the chat history
        read_chat_history=True,
    )
    if run_id is None:
        run_id = assistant.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    assistant.cli_app(markdown=True)

if __name__=="__main__":
    typer.run(rag_agent)
