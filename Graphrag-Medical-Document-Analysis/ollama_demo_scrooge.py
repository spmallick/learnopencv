import instructor  # Structured outputs for LLMs

from fast_graphrag import GraphRAG
from fast_graphrag._llm import OpenAIEmbeddingService, OpenAILLMService


DOMAIN = "Analyze this story and identify the characters. Focus on how they interact with each other, the locations they explore, and their relationships."

EXAMPLE_QUERIES = [
    "What is the significance of Christmas Eve in A Christmas Carol?",
    "How does the setting of Victorian London contribute to the story's themes?",
    "Describe the chain of events that leads to Scrooge's transformation.",
    "How does Dickens use the different spirits (Past, Present, and Future) to guide Scrooge?",
    "Why does Dickens choose to divide the story into \"staves\" rather than chapters?"
]

ENTITY_TYPES = ["Character", "Animal", "Place", "Object", "Activity", "Event"]


working_dir = "./WORKING_DIR/carol/test"

grag = GraphRAG(
    working_dir=working_dir,
    # n_checkpoints=2,
    domain = DOMAIN,
    example_queries="\n".join(EXAMPLE_QUERIES),
    entity_types=ENTITY_TYPES,
    config=GraphRAG.Config(
        llm_service=OpenAILLMService(
            model = "llama3.1:8b",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            mode = instructor.Mode.JSON,
            client="openai",
            
        ),
        
#         llm_service=OpenAILLMService(
#             model = "gemini-2.0-flash-lite-preview-02-05", # 
#             base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
#             api_key="AIzaSyBDcIHwi4TX7MOMQ_Nnd6AKz8j5KxdGa7o",
#             mode = instructor.Mode.JSON,
           
#             # client="openai"
#    ),

        
        
        embedding_service=OpenAIEmbeddingService(
            model = "mxbai-embed-large" , # mxbai-embed-large
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            embedding_dim=1024,  # for mxbai-embed-large - 1024
            # client="openai"
            
        )
        
    )
)


with open("./book.txt") as f:
    grag.insert(f.read())
    
print("**********************************************")

print(grag.query("Who is Scrooge?").response)

print("**********************************************")
print(grag.query("List all the characters?").response)

print("**********************************************")

print(grag.query("What are the odd  events in the story?").response)

print("**********************************************")

print(grag.query("What is the overall theme of the story?").response)
