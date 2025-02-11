from crewai import Task
from tools import tool
from services import article_researcher,article_writer

# Research task
research_task = Task(
    description=(
        "Identify the most relevant articles about the {topic} from learnopencv.com."
        "Create a well-structured roadmap for a beginner learner so that he/she can go" 
        "through all individual articles one-by-one to completely know about the topic from" 
        "scratch till advanced knowledge about the topic"
        "Include the article title, date and authors and co-authors for all the articles found"
        "on learnopencv.com about the topic for creating the roadmap."
        "Make sure to not to repeat any article in the roadmap."
    ),
    expected_output='List of all the relevant articles with the article titles, published date and all authors and co-authors information in order so that a beginner learner can go through all articles order-wise to learn about the topic from scratch till advanced knowledge.',
    tools=[tool],
    agent=article_researcher,
    sources=["https://learnopencv.com"]  # Restricting search to learnopencv.com
)

# Writing task with language model configuration
write_task = Task(
    description=(
        "Compose a well-structured roadmap for the {topic}."
        "Focus on all important articles related to the topic."
        "Extract the meta-description of all the articles found and then arrange the articles in a" 
        "way so that the beginner learner can start learning about the topic by starting from the start of the roadmap."
        "Ensure that the roadmap must include all author and co-author details details and date of publication for all the" 
        "articles to be put in the roadmap found from learnopencv.com."
        "Make sure to include the names of all the authors and co-authors and publication dates too."
        "The roadmap must be well-structured and organised in a manner such that the beginner learner can directly look at it" 
        "and then can directly look for the article which he/she wants to look upon."
    ),
    expected_output='A well-structured roadmap for a beginner learner to refer, on the {topic} found on learnopencv.com and the roadmap must also include article titles, date and author information of all the articles to be structured in the roadmap.',
    tools=[tool],
    agent=article_writer,
    async_execution=False,
    output_file='Article.md'  # Example of output customization
)
