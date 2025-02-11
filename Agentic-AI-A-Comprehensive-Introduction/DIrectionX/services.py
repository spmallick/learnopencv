from crewai import Agent
from tools import tool
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
import os


## call the gemini models
llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp",
                           verbose=True,
                           temperature=0.5,
                           google_api_key=os.getenv("GEMINI_API_KEY"))

# Creating a Research agent with memory and verbose mode

article_researcher = Agent(
    role="Researcher",
    goal="Search for the articles related to the {topic} by searching exclusively on 'learnopencv.com'." 
        "And then you need to make the list of all relevant articles found about the topic and then make a list"
        "of all those articles alongwith the article titles, names of all authors and co-authors of the respective" 
        "articles and also include the date of publication of those articles. The arrangement of articles should be" 
        "in such a way that a beginner learner can refer to it and then start learning about the articles order-wise" 
        "to learn about the topic from scratch till advanced knowledge.",
    verbose=True,
    memory=True,
    backstory=(
        "You're at the forefront of AI and Computer Vision research."
        "Your expertise is in searching the most relevant articles about the topic and make a list out of them."
        "Our primary focus is identifying the most relevant article from 'learnopencv.com'."
        "Extract and provide the article titles, publication date and the names of all the authors and co-authors "
        "of all the relevant articles found from learnopencv.com."
        "Make the list of all the relevant articles in such a way that they are arranged to create a well-structured" 
        "roadmap out of it so that a beginner learner can refer to the roadmap and then start learning about the topic"
        "by referring the roadmap from beginning to the end."
    ),
    tools=[tool],
    llm=llm,
    allow_delegation=True
)


## Creating the article writer agent with tools responsible in writing the final article

article_writer = Agent(
    role="Writer",
    goal="Generate the well-structured and organized roadmap about the {topic} by evaluating the meta-descriptions" 
         "of the most relevant articles found on learnopencv.com like which article to be put first and which article"
         "after the another one so that it will be a roadmap for a beginner learner to start learning about the topic" 
         "by referring the roadmap from beginning to the end.",
    verbose=True,
    memory=True,
    backstory=(
        "You generate the final well-structured and organized roadmap in the most professional way."
        "Your generated roadmap should be insightful and must be based on "
        "the best-matching articles about the topic from learnopencv.com and the order of the articles"
        "in the roadmap must be in such a way that the first article will be that of an introductory article on the" 
        "topic and then gradually the level of the articles increases."
        "The order of the articles to be structured in the roadmap must be in such a way that the beginner learner" 
        "will start going through all the articles mentioned in the roadmap order-wise and he/she will be able to learn"
        "everything about the topic from scratch till advanced knowledge as he/she continues with the roadmap."
        "Ensure the roadmap must include the exact titles of the articles, their publication dates and the names of all" 
        "the authors and co-authors of the articles found from learnopencv.com."
    ),
    tools=[tool],
    llm=llm,
    allow_delegation=False
)

