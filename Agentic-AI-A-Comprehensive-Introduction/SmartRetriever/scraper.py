from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from crewai_tools import ScrapeWebsiteTool
import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
## call the gemini models
llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp",
                           verbose=True,
                           temperature=0.5,
                           google_api_key=os.getenv("GEMINI_API_KEY"))

# Instantiate tools
site = 'https://docs.opencv.org/3.4/d9/d25/group__surface__matching.html'
web_scrape_tool = ScrapeWebsiteTool(website_url=site)


# Create agents
web_scraper_agent = Agent(
    role='Web Scraper',
    goal='Effectively Scrape data on the websites for your company',
    backstory='''You are expert web scraper, your job is to scrape all the data for 
                your company from a given website.
                ''',
    tools=[web_scrape_tool],
    verbose=True,
    llm = llm
)


# Define tasks
web_scraper_task = Task(
    description='Scrape all the  data on the site so your company can use for decision making.',
    expected_output='All the content of the website.',
    agent=web_scraper_agent,
    output_file = 'data.txt'
)


# Assemble a crew
crew = Crew(
    agents=[web_scraper_agent],
    tasks=[web_scraper_task],
    verbose=True, 
)

# Execute tasks
result = crew.kickoff()
print(result)

with open('results.txt', 'w') as f:
    f.write(result)