from dotenv import load_dotenv
from pydantic import BaseModel
from bs4 import BeautifulSoup as soup
from langgraph.graph import StateGraph, END, START
from langgraph.graph import add_messages
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.output_parsers import PydanticOutputParser

load_dotenv()

# === Load and preprocess documentation ===
url = 'https://huggingface.co/docs/diffusers/stable_diffusion'
loader = RecursiveUrlLoader(
    url=url,
    max_depth=20,
    extractor=lambda x: soup(x, 'html.parser').text
)
docs = loader.load()
d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join([doc.page_content for doc in d_reversed])

# === Define structured output ===
class DiffuserCodeOutput(BaseModel):
    description: str
    code: str
    explanation: str

parser = PydanticOutputParser(pydantic_object=DiffuserCodeOutput)

# === LLM Initialization ===
llm = GoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

# === Agent State ===
class AgentState(TypedDict):
    error: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    output: str
    description: str
    explanation: str
    iterations: int

# === System Prompt ===
system_message = SystemMessage(
    content=f"""<instructions>
You are an expert coding assistant specializing in the Hugging Face `diffusers` library.
Your task is to answer the user's question by generating a complete, executable Python script.

Here is the relevant `diffusers` documentation to help you:
-------
{concatenated_content}
-------

You must respond in a **JSON format** with the following fields:
{parser.get_format_instructions()}

Strictly follow this structure:
1. `description`: A one-line summary of what the script does.
2. `code`: A full working Python script (no markdown formatting).
3. `explanation`: A short paragraph explaining key parameters and decisions.
</instructions>"""
)

# === Agent Functions ===
max_iterations = 5
flag = "do not reflect"

def generate(state: AgentState) -> AgentState:
    print('Generating code solution')

    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]

    if error == "yes":
        messages.append(HumanMessage(content="Try again. Fix the code and follow the structure."))

    response_text = llm.invoke(messages)
    parsed = parser.parse(response_text)

    messages.append(AIMessage(content=response_text))
    iterations += 1

    return {
        "output": parsed.code,
        "description": parsed.description,
        "explanation": parsed.explanation,
        "iterations": iterations,
        "messages": messages,
        "error": ""
    }

def code_check(state: AgentState) -> AgentState:
    print("Code Checking")

    messages = state["messages"]
    iterations = state["iterations"]
    code = state["output"]

    try:
        exec(code, globals())
    except Exception as e:
        print("Code execution failed:", e)
        messages.append(HumanMessage(content=f"Your solution failed: {e}"))
        return {
            "output": code,
            "description": state["description"],
            "explanation": state["explanation"],
            "messages": messages,
            "iterations": iterations,
            "error": "yes"
        }

    print("Code executed successfully")
    return {
        "output": code,
        "description": state["description"],
        "explanation": state["explanation"],
        "messages": messages,
        "iterations": iterations,
        "error": "no"
    }

def reflect(state: AgentState) -> AgentState:
    print("Reflecting on the error")

    messages = state["messages"]
    iterations = state["iterations"]

    messages.append(HumanMessage(content="Reflect on the error and try again."))
    reflection = llm.invoke(messages)
    messages.append(AIMessage(content=reflection))

    return {
        "output": state["output"],
        "description": state["description"],
        "explanation": state["explanation"],
        "messages": messages,
        "iterations": iterations,
        "error": state["error"]
    }

def should_continue(state: AgentState) -> str:
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations >= max_iterations:
        print("Decision: Finish")
        return "end"
    else:
        print("Decision: Re-Try solution")
        return "reflect" if flag == "reflect" else "generate"

# === LangGraph Workflow ===
workflow = StateGraph(AgentState)

workflow.add_node("generate", generate)
workflow.add_node("check_code", code_check)
workflow.add_node("reflect", reflect)

workflow.add_edge(START, "generate")
workflow.add_edge("generate", "check_code")
workflow.add_conditional_edges("check_code", should_continue, {
    "end": END,
    "reflect": "reflect",
    "generate": "generate"
})
workflow.add_edge("reflect", "generate")

app = workflow.compile()

# === Run the Agent ===
initial_question = "generate image of an old man in 20 inference steps"

initial_state = {
    "messages": [
        system_message,
        HumanMessage(content=initial_question)
    ],
    "iterations": 0,
    "error": "",
    "output": "",
    "description": "",
    "explanation": "",
}

solution = app.invoke(initial_state)

# === Final Output ===
print("\n--- FINAL RESULT ---")
print("📝 Description:\n", solution["description"])
print("\n📄 Code:\n", solution["output"])
print("\n🔍 Explanation:\n", solution["explanation"])