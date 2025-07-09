import base64, asyncio
from dotenv import load_dotenv
from typing import Annotated, Sequence, List, TypedDict, Union

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

from playwright.async_api import async_playwright, Page, Browser


load_dotenv()

#another way of implementing browser: Browser | None
browser = Union[Browser, None] 
page = Union[Page, None]


#defining the state dictionary which will be passed between nodes
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] #for conversation history
    url: Union[str, None]
    current_ss: Union[List[str], None]
    summaries: Annotated[Sequence[BaseMessage], add_messages]
    scroll_decision: Union[str, None]
    task: str


async def initialize_browser():
    """
    Initialize the playwright browser and page
    """

    global browser, page
    print('-----Initializing the Playwright browser-----')

    try:
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless = False)

        page = await browser.new_page()
        print('-----Browser Initialized-----')

    except Exception as e:
        print(f'Failed to initialized browser due the following exception: {e}')

async def close_browser():
    """
    Closes the Playwright Browser.
    """

    global browser, page

    if browser:
        print('-----Closing the Playwright browser-----')
        try:
            await browser.close()
            print('-----Browser Closed-----')
        except Exception as e:
            print(f'Error in closing the browser:{e}')

        finally:
            browser = None
            page = None


@tool
async def navigate_url(url: str) -> str:
    """
    This tool takes the browser to navigate to the URL provided via Playwright.
    """
    global page
    print('-----Navigating to the provided URL-----')
    try: 
        await page.goto(url, wait_until = 'domcontentloaded')  
        # await asyncio.sleep(2)
        return f'-----Successfully Navigated-----'  
      
    except Exception as e:
        return f'The Error that occured during navigating url is:{e}'


#now, we will define tools using langchain's provided function tools as a decorator
#return type is string because we are using base64
#This module provides functions for encoding binary data to printable ASCII 
#characters and decoding such encodings back to binary data
@tool
async def take_ss() -> str:
    """
    Takes screenshot of the current browser state via Playwright.
    """

    global page

    if page is None:
        return '-----Browser page not initialized-----'
    
    else:
        print('*****ACTION: taking screenshot of the current browser state*****') 

        try:
            binary_ss = await page.screenshot()
            b64_ss = base64.b64encode(binary_ss).decode("utf-8")

            print('-----Screenshot successfully captured-----')
            return b64_ss
        
        except Exception as e:
            return f'Error that occured during taking screenshot:{e}' 


@tool
async def scroll_down() -> str:
    """
    Scrolls the page down by a fixed amount.
    """

    global page

    if page is None:
        return "-----Page not initialised-----"

    viewport_height = await page.evaluate("window.innerHeight")
    scroll_amount   = int(viewport_height * 0.8)

    await page.evaluate(f"window.scrollBy(0, {scroll_amount});")
    # await asyncio.sleep(0.5)  # small wait

    return f"*****Scrolled {scroll_amount}px*****"

agent_tools = [navigate_url, take_ss, scroll_down]
llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash').bind_tools(tools = agent_tools)



async def init_node(state: AgentState) -> AgentState:
    """
    Initializes the browser and navigates to the provided URL.
    """

    print('-----Initial Node-----')
    task = state['task']

    base_url = "https://en.wikipedia.org/wiki/Large_language_model"

    await initialize_browser()
    navigate_output = await navigate_url.ainvoke(base_url)

    return {
        **state,
        'url': base_url,
        'messages': [SystemMessage(content=f'Navigated to the provided URL:{base_url}. {navigate_output}')]
    }

async def ss_node(state: AgentState) -> AgentState:
    """
    Takes a screenshot of the current page using the take_ss tool
    and stores it in the state as a list.
    """

    print('-----Screenshot Node-----')
    try:
        b64_ss = await take_ss.ainvoke() #input = None removed
        print("*****Screenshot captured and returned from tool*****")

        current_ss_list = state.get("current_ss") #.get method of dictionaries to access values corresponding to a key.
        if current_ss_list is None:
            current_ss_list = []

        current_ss_list.append(b64_ss)

        # updated_messages = state.get("messages") + [
        #     SystemMessage(content="Screenshot captured and saved to state.")
        # ]

        updated_messages = [SystemMessage(content= "Screenshot captured and saved to state variable.")]

        return {
            **state,
            "current_ss": current_ss_list,
            "messages": updated_messages,
        }

    except Exception as e:
        error_msg = f"Error during ss_node: {e}"
        print(error_msg)
        return {
            **state,
            # "messages": state.get("messages", []) + [SystemMessage(content=error_msg)],
            "messages": [SystemMessage(content = error_msg)]
        }



async def summarizer_node(state: AgentState) -> AgentState:
    """
    Uses the LLM to summarize the current screenshot and page state.
    The latest screenshot is sent as a base64 image to the model.
    """

    print("-----Summarizer Node-----")
    task = state.get("task", "Summarize this page as briefly as possible")
    screenshots = state.get("current_ss")

    if not screenshots:
        print("-----No screenshot available to summarize-----")
        return {
            **state,
            "summaries": [SystemMessage(content= "No screenshot available for summarization")]
            # "summaries": state.get("summaries", []) + [SystemMessage(content="No screenshot available for summarization.")]
        }

    latest_ss = screenshots[-1]  # Only use the most recent one

    user_prompt = HumanMessage(content=[
        {"type": "text", "text": f"Summarize this screenshot for the following task:{task}"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{latest_ss}"}}
    ])

    try:
        summary = await llm.ainvoke([user_prompt])
        print("*****LLM summarization successful*****")

        return {
            **state,
            "summaries":[summary],
            "messages": [SystemMessage(content="Page summary generated.")]
            # "summaries": state.get("summaries", []) + [summary],
            # "messages": state.get("messages", []) + [SystemMessage(content="Page summary generated.")],
        }

    except Exception as e:
        error_msg = f"Error during summarization: {e}"
        print(error_msg)
        return {
            **state,
            "messages": [SystemMessage(content=error_msg)]
            # "messages": state.get("messages", []) + [SystemMessage(content=error_msg)],
        }




async def scroll_decision_node(state: AgentState) -> AgentState:
    """
    Calls the scroll_down tool *and waits for it*.
    Logs the before/after scroll positions for debugging.
    """
    global page
    if page is None:
        return {**state,
                "messages": state["messages"]
                + [SystemMessage(content="Scroll skipped – page not initialised.")]}
    
    # How far down are we now?
    before = await page.evaluate("window.scrollY")

    tool_result = await scroll_down.ainvoke(input=None)

    after  = await page.evaluate("window.scrollY")
    moved  = after - before

    return {**state,
            "messages": state["messages"]
            + [SystemMessage(content=f"{tool_result}  (Δy = {moved}px)")]}




async def aggregate_node(state: AgentState) -> AgentState:
    """
    Aggregates all summaries into a final report.
    """
    print("Aggregation Node-----")
    summaries = state.get("summaries", [])
    task = state.get("task", "")

    messages = [
        SystemMessage(content=f"Aggregate the following summaries for the task: {task}"),
        HumanMessage(content="\n\n".join([msg.content for msg in summaries if hasattr(msg, "content")]))
    ]

    try:
        final_summary = await llm.ainvoke(messages)
        return {
            **state,
            "messages": state["messages"] + [SystemMessage(content="Final summary created."), final_summary],
        }

    except Exception as e:
        return {
            **state,
            "messages": state["messages"] + [SystemMessage(content=f"Error during aggregation: {e}")],
        }


def route_scroll_decision(state: AgentState) -> str:
    """
    Routes the next step based on scroll decision ('yes'/'no').
    Also routes to 'aggregate' if browser initialization failed.

    !!! MODIFIED TO ALWAYS RETURN "scroll" FOR TESTING !!!
    """
    print("Routing based on scroll decision (forced scroll)...")

    # We still check if browser initialization failed, as we cannot scroll in that case
    messages = state.get("messages", [])
    init_failed = any("Browser initialization failed." in msg.content for msg in messages if isinstance(msg, SystemMessage))

    if init_failed:
        print("Browser initialization failed, routing to aggregate.")
        return "aggregate"

    # Force the routing to the 'scroll' node for testing the scroll tool
    # You should revert this change to use the LLM's decision for normal operation
    print("Forcing route to 'scroll' node.")
    return "scroll"


workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("init", init_node)
workflow.add_node("screenshot", ss_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("decide_scroll", scroll_decision_node)
workflow.add_node("scroll", lambda state: scroll_down.ainvoke(input=None).then(lambda _: state))
workflow.add_node("aggregate", aggregate_node)

# Entry
workflow.set_entry_point("init")

# Flow
workflow.add_edge("init", "screenshot")
workflow.add_edge("screenshot", "summarizer")
workflow.add_edge("summarizer", "decide_scroll")

workflow.add_conditional_edges(
    "decide_scroll",
    route_scroll_decision,
    {
        "scroll": "screenshot",     # loop
        "aggregate": "aggregate"    # exit
    }
)

workflow.add_edge("aggregate", END)

app = workflow.compile()



if __name__ == "__main__":
    initial_state = {
        "messages": [],
        "url": None,
        "current_ss": [],
        "summaries": [],
        "scroll_decision": None,
        "task": "Give a brief overview of this Wikipedia page on Large Language Models."
    }

    print("\n--- Starting LangGraph Agent ---\n")

async def run_graph():
    initial_state = {
        "messages": [],
        "url": None,
        "current_ss": [], # Initialize as empty list to store base64 strings
        "summaries": [],  # Initialize as empty list to store summary messages
        "scroll_decision": None,
        "task": "Give a brief overview of this Wikipedia page on Large Language Models." # Example task
    }
    print("\n--- Starting LangGraph Agent ---\n")

    # Wrap the astream in a try/finally to ensure browser closure even on error
    try:
        # Use astream to see the state changes step-by-step
        # Set recursion_limit to prevent infinite loops in case of unexpected behavior
        async for step in app.astream(initial_state, {"recursion_limit": 20}): # Limit to 10 steps to prevent infinite loops
            step_name = list(step.keys())[0]
            print(f"\n--- Step: {step_name} ---")

            # Access the state *after* the node execution
            latest_state = step[step_name]

            # Print specific information based on the node that just completed
            if step_name == "summarizer":
                # Find the latest summary message added by the summarizer
                # Check the summaries list, it should contain the latest summary from the LLM
                if latest_state.get('summaries'):
                    # The latest summary message is the last one added
                    latest_summary_message = latest_state['summaries'][-1]
                    if isinstance(latest_summary_message, (AIMessage, HumanMessage)) and latest_summary_message.content:
                         print(">>> Individual Screenshot Summary:")
                         print(latest_summary_message.content)
                    elif isinstance(latest_summary_message, SystemMessage):
                         # Print messages like "No screenshot available"
                         print(">>> Summarizer Status:", latest_summary_message.content)


            elif step_name == "decide_scroll":
                # Print the scroll decision
                decision = latest_state.get('scroll_decision')
                print(f">>> Scroll Decision: {decision}")

            elif step_name == "aggregate":
                # The aggregation node adds the final summary as a HumanMessage to the messages list
                # Look for the latest HumanMessage in the messages list that seems like the final summary
                final_summary_message = None
                # Iterate backwards through messages to find the latest summary-like message
                for msg in reversed(latest_state.get('messages', [])):
                    # Heuristic: The aggregation node adds a SystemMessage "Final summary created." just before the HumanMessage
                    if isinstance(msg, HumanMessage) and final_summary_message is None:
                         final_summary_message = msg # Potential final summary
                    elif isinstance(msg, SystemMessage) and msg.content == "Final summary created." and final_summary_message is not None:
                         # Found the system message preceding a potential summary, confirm it
                         print(">>> Final Aggregated Summary:")
                         print(final_summary_message.content)
                         break # Found and printed, exit loop

                # Fallback in case the heuristic fails or no valid summary was produced
                if final_summary_message is None:
                     print(">>> Aggregation Node Finished (No valid final summary found in messages).")
                     # You could add logic here to print a specific error message from state['messages'] if aggregation failed


    except Exception as e:
        print(f"\n--- An error occurred during graph execution: {e} ---")
    finally:
        print("\n--- Agent execution finished. Attempting to close browser. ---")
        # Ensure the browser is closed even if the graph execution fails
        await close_browser()


if __name__ == "__main__":
    # asyncio.run() is needed to run async functions at the top level
    # This runs the entire async execution including browser initialization and closure
    asyncio.run(run_graph())