from crewai import Crew,Process
from tasks import research_task,write_task
from services import article_researcher,article_writer

## Declaring Crew to Streamline the Execution Process
crew=Crew(
    agents=[article_researcher,article_writer],
    tasks=[research_task,write_task],
    process=Process.sequential,

)

## Starting the Task Execution process 
result=crew.kickoff(inputs={'topic':'YOLO'})
print(result)