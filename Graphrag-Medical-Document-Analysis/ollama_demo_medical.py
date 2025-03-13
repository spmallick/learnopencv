import instructor  # Structured outputs for LLMs
import os
from fast_graphrag import GraphRAG, QueryParam
from fast_graphrag._llm import OpenAIEmbeddingService, OpenAILLMService


DOMAIN = "Analyze these clinical records and identify key medical entities. Focus on patient demographics, diagnoses, procedures, lab results, and outcomes."

EXAMPLE_QUERIES = [
    "What are the common risk factors for sepsis in ICU patients?",
    "How do trends in lab results correlate with patient outcomes in cases of acute kidney injury?",
    "Describe the sequence of interventions for patients undergoing major cardiac surgery.",
    "How do patient demographics and comorbidities influence treatment decisions in the ICU?",
    "What patterns of medication usage are observed among patients with chronic obstructive pulmonary disease (COPD)?"
]

ENTITY_TYPES = ["Patient", "Diagnosis", "Procedure", "Lab Test", "Medication", "Outcome"]


working_dir = "./WORKING_DIR/mimic_ex500/"

grag = GraphRAG(
    working_dir=working_dir,
    n_checkpoints=2,
    domain=DOMAIN,
    example_queries="\n".join(EXAMPLE_QUERIES),
    entity_types=ENTITY_TYPES,
    config=GraphRAG.Config(
        # ****** OPENAI COMPATIBLE ENDPOINTS *************
        llm_service=OpenAILLMService(
            model="Phi4_6k",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            mode=instructor.Mode.JSON,
            client="openai",
        ),
        # **************** GEMINI FLASH **************************
        #         llm_service=OpenAILLMService(
        #             model = "gemini-2.0-flash-lite-preview-02-05", #
        #             base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        #             api_key="AIzaSyBDcIHwi4TX7MOMQ_Nnd6AKz8j5KxdGa7o",
        #             mode = instructor.Mode.JSON,
        #             # client="openai"
        #    ),
        # *******************************************************
        
        embedding_service=OpenAIEmbeddingService(
            model="mxbai-embed-large",  # mxbai-embed-large
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            embedding_dim=1024,  # for mxbai-embed-large - 1024
            client="openai"
        ),
    ),
)

# Download dataset: huggingface.co/datasets/morson/mimic_ex/blob/main/dataset.zip


directory_path = "mimic_ex_500"

def graph_index(directory_path):
    file_count = 0 # Keep track of processed files.
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                content = file.read()
                
                # Convert list to string if needed
                    # If content follows a nested structure like {"text": "actual content"}
                if isinstance(content, list):
                    content = "\n".join(map(str, content))
                if isinstance(content, dict):
                    key_to_use = next(iter(content.keys()), None)
                    content = content[key_to_use] if key_to_use else str(content)
                
                else:
                    content = str(content)   
                
                
                grag.insert(content)
            
            file_count += 1
     
            total_files = sum(1 for f in os.listdir(directory_path) if f.endswith(".txt"))
            print("******************** $$$$$$ *****************")        
            print(f"Total Files Processed: -> {file_count} / {total_files}")
            print("******************** $$$$$$ *****************")  
    return None

graph_index(directory_path)


print("**********************************************")

os.makedirs("neo4j_graph", exist_ok=True)
grag.save_graphml(output_path="neo4j_graph/oxford_graph_chunk_entity_relation.graphml")

print(grag.query("Provide indepth detail about In patients with both chronic obstructive pulmonary disease (COPD) and heart failure, how can lung function be improved?").response)


# ****** More on this: https://github.com/circlemind-ai/fast-graphrag/blob/main/examples/query_parameters.ipynb ************

# Querying with references
# query ="Discuss about end-stage renal disease (ESRD)"
# answer = grag.query(query, QueryParam(with_references=True))

# print(answer.response)  # ""
# print(answer.context)  # {entities: [...], relations: [...], chunks: [...]}
# print(answer.response)



# Get top 10 entities
#print(answer.context.entities[:10])  # {entities: [...], relations: [...], chunks: [...]}


# Format response
# fresponse, fresponse = answer.format_references(
#     lambda doc_index, chunk_indices, metadata: f"[{doc_index}.{'-'.join(map(str, chunk_indices))}]({metadata['id']})"
# )


