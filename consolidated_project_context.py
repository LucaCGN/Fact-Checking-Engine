#
# consolidated_project_context.py
#


#
# gen_context.py
#
import os
import re

project_dir = '.'
consolidated_file = 'consolidated_project_context.py'

with open(consolidated_file, 'w') as outfile:
  for root, dirs, files in os.walk(project_dir):
    for file in files:
      if file.endswith('.py'):
        filepath = os.path.join(root, file)
        relpath = os.path.relpath(filepath, project_dir)

        # Write header with relative path
        outfile.write(f'#\n# {relpath}\n#\n')

        # Read and write content, removing any trailing newline
        with open(filepath) as infile:
          content = infile.read().rstrip('\n')  # Remove trailing newline
          outfile.write(content)
          outfile.write('\n\n')  # Add blank lines for separation

#
# main.py
#


#
# agents\journalist.py
#
"""
Processes an image description asynchronously using an LLM to generate a list of points that can be fact-checked.
"""
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import asyncio

class FactCheckingProcessor:
    def __init__(self):
        self.llm = OpenAI(model="gpt-4", temperature=0.5)  # Instantiate GPT-4 LLM with desired parameters

    async def process_description(self, description: str) -> list:
        """
        Asynchronously processes the image description and returns a list of points for fact-checking.
        """
        prompt_template = PromptTemplate.from_template(
            "Given the description '{}', as a journalist seeking the truth, "
            "list out specific points that can be fact-checked to determine the veracity of the description."
        )

        prompt = prompt_template.render(description=description)

        response = await self.llm.invoke_async(prompt)

        fact_checking_points = self._parse_response_into_points(response)

        return fact_checking_points

    def _parse_response_into_points(self, response: str) -> list:
        """
        Parses the LLM response into a list of discrete points.
        Extracts points structured as individual questions or statements for fact-checking.
        """
        points = response.split("\n")
        return [point.strip() for point in points if point.strip()]

# Example usage
if __name__ == "__main__":
    fact_checker = FactCheckingProcessor()
    description = "An image showing Donald Trump being arrested."

    async def run():
        points_to_fact_check = await fact_checker.process_description(description)
        print(points_to_fact_check)

    asyncio.run(run())

#
# agents\researcher.py
#
import luigi
import json
import asyncio
from aiohttp import ClientSession
from tools import factcheck_api, newsapi, serpapi
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

class ResearcherConsolidationTask(luigi.Task):
    """
    Consolidates information from various sources, including direct web searches, news articles,
    and fact-checking databases. It uses LangChain to enhance data analysis and draw meaningful insights.
    """
    search_queries = luigi.ListParameter()

    async def fetch_data(self, url, session):
        """
        Asynchronously fetches data from a given URL using aiohttp.
        """
        async with session.get(url) as response:
            return await response.json()

    async def gather_information(self, queries):
        """
        Uses LangChain and other APIs to gather and analyze information based on search queries.
        """
        async with ClientSession() as session:
            responses = await asyncio.gather(
                *(self.fetch_data(factcheck_api.search_url(query), session) for query in queries),
                *(self.fetch_data(newsapi.search_url(query), session) for query in queries),
                *(self.fetch_data(serpapi.search_url(query), session) for query in queries)
            )

            # Initialize LangChain OpenAI model for analysis
            llm = OpenAI(model="gpt-4", temperature=0.5)
            prompt_template = PromptTemplate.from_template(
                "Given the following information '{information}', summarize key insights and potential areas for further investigation."
            )

            insights = []
            for response in responses:
                prompt = prompt_template.render(information=json.dumps(response, ensure_ascii=False))
                insight = await llm.invoke_async(prompt)
                insights.append(insight)

            return insights

    def run(self):
        """
        Orchestrates the task of gathering information from multiple sources and consolidating it into meaningful insights.
        """
        loop = asyncio.get_event_loop()
        insights = loop.run_until_complete(self.gather_information(self.search_queries))

        # Consolidate insights and write to output
        with self.output().open('w') as f:
            json.dump(insights, f)

    def output(self):
        """
        Specifies the output file for the consolidated insights.
        """
        return luigi.LocalTarget("consolidated_insights.json")

if __name__ == "__main__":
    # Example usage
    search_queries = ["climate change effects", "latest technology trends"]
    luigi.build([ResearcherConsolidationTask(search_queries=search_queries)], local_scheduler=True)

#
# agents\transcriber.py
#
"""
Transcribes an audio file asynchronously using OpenAI's Whisper API.

Orchestrates reading the audio file, sending the audio data to OpenAI, 
and writing the transcription result to a file.

Returns the transcription text on success, error message on failure.
"""
import luigi
import os
import asyncio
from aiohttp import ClientSession
from getpass import getpass

# Prompt the user for the OpenAI API token and set it as an environment variable
OPENAI_API_TOKEN = getpass("Enter your OpenAI API token: ")
os.environ["OPENAI_API_TOKEN"] = OPENAI_API_TOKEN

class TranscribeAudioTask(luigi.Task):
    # The path for the audio file to be transcribed
    audio_path = luigi.Parameter()

    def run(self):
        """
        Orchestrates the process of transcribing an audio file using the Whisper model.
        This involves asynchronous I/O operations, including reading the audio,
        sending a request to the OpenAI API, and writing the output to a file.
        """
        loop = asyncio.get_event_loop()
        transcription = loop.run_until_complete(self.transcribe_audio())
        self.set_status_message(transcription)
        self.output().write(transcription)

    async def transcribe_audio(self):
        # Create an asynchronous HTTP session
        async with ClientSession() as session:
            # Open the audio file in binary read mode
            with open(self.audio_path, "rb") as audio_file:
                audio_content = audio_file.read()

            # Construct the payload for the POST request to the OpenAI API
            payload = {
                "model": "whisper",
                "audio": audio_content,
                # Additional parameters might be required depending on the API
            }

            try:
                # Execute the POST request to start the transcription process
                async with session.post("https://api.openai.com/v1/transcriptions", json=payload, headers={"Authorization": f"Bearer {OPENAI_API_TOKEN}"}) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        # Success: Extract and return the transcription from the response
                        return response_data.get("transcription")
                    else:
                        # Handle HTTP errors by returning an error message
                        return "Error: Failed to transcribe audio."
            except Exception as e:
                # Handle exceptions by returning an error message
                return f"Error: {str(e)}"

    def output(self):
        # Define the output file path (adding "_transcription.txt" to the audio path)
        return luigi.LocalTarget(self.audio_path + "_transcription.txt")

if __name__ == "__main__":
    # Example: Specify the path of the audio to be transcribed
    audio_path = "test_input/test_audio.wav"
    # Start the Luigi pipeline with the transcribe audio task
    luigi.build([TranscribeAudioTask(audio_path=audio_path)], local_scheduler=True)

#
# agents\vision.py
#
"""
Generates a description of an image using the Replicate API. 

Downloads the image, constructs a prompt requesting a detailed description, 
calls the Replicate API to generate the description, and writes the result to a file.
"""
import luigi
import replicate
import os
import asyncio
from aiohttp import ClientSession
from getpass import getpass

# Prompt the user for the Replicate API token and set it as an environment variable
REPLICATE_API_TOKEN = getpass("Enter your Replicate API token: ")
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

class DescribeImageTask(luigi.Task):
    # The path for the image to be described
    image_path = luigi.Parameter()

    def run(self):
        """
        Orchestrates the process of generating a description for an image.
        This involves asynchronous I/O operations, including reading the image,
        sending a request to the Replicate API, and writing the output to a file.
        """
        loop = asyncio.get_event_loop()
        description = loop.run_until_complete(self.describe_image())
        self.set_status_message(description)
        self.output().write(description)

    async def describe_image(self):
        # Create an asynchronous HTTP session
        async with ClientSession() as session:
            # Define the prompt for the image description task
            prompt = "Please provide a detailed and assertive description of the image. " \
                     "Specifically identify any public figures present in the image."
            # Open the image file in binary read mode
            with open(self.image_path, "rb") as image_file:
                image_content = image_file.read()

            # Construct the payload for the POST request to the Replicate API
            payload = {
                "image": image_content,
                "prompt": prompt
            }

            try:
                # Execute the POST request to start the prediction process
                async with session.post("https://api.replicate.com/v1/predictions", json=payload) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        # Success: Extract and return the description from the response
                        return response_data.get("output")
                    else:
                        # Handle HTTP errors by returning an error message
                        return "Error: Failed to generate description."
            except Exception as e:
                # Handle exceptions by returning an error message
                return f"Error: {str(e)}"

    def output(self):
        # Define the output file path (adding "_description.txt" to the image path)
        return luigi.LocalTarget(self.image_path + "_description.txt")

if __name__ == "__main__":
    # Example: Specify the path of the image to be described
    image_path = "test_input/test_trump.jpeg"
    # Start the Luigi pipeline with the described image task
    luigi.build([DescribeImageTask(image_path=image_path)], local_scheduler=True)

#
# agents\__init__.py
#


#
# pipelines\deepfake_audio.py
#
# call tools\audio_formatting.py to stardanrize input.

# call agents\transcriber.py to give meaning and context to the transcription.

# call agents\journalist.py to create factchecking plan.

# call   tools\factcheck_api.py
         #tools\newsapi.py
         #tools\serpapi.py

        # Perform Searchs and Output Documentation Files on the subject.

# pass output to agents\researcher.py

# pass output to agents\journalist.py

#
# pipelines\deepfake_image.py
#
# call tools\image_formatting.py to stardanrize input.

# call agents\vision.py to give meaning and context to the image.

# call agents\journalist.py to create factchecking plan.

# call   tools\factcheck_api.py
         #tools\newsapi.py
         #tools\serpapi.py

        # Perform Searchs and Output Documentation Files on the subject.

# pass output to agents\researcher.py

# pass output to agents\journalist.py

#
# pipelines\source_crossing.py
#


#
# pipelines\statement_check.py
#


#
# tools\audio_formatting.py
#


#
# tools\factcheck_api.py
#
from langchain.llms import OpenAI
from langchain.chains import create_openai_function_chain
from langchain.schema import BasePromptTemplate, StrOutputParser, FunctionSchema
import json
import asyncio

class FactCheckAPITool:
    def __init__(self, openai_api_key: str, model: str = "gpt-3.5-turbo"):
        # Initialize the OpenAI model with the provided API key
        self.llm = OpenAI(api_key=openai_api_key, model=model)

    async def search_claims(self, query: str, language_code: str = "en", review_publisher_site_filter: str = None,
                            max_age_days: int = None, page_size: int = 10, page_token: str = None):
        # Define the API call parameters as part of the function schema
        function_schema = FunctionSchema(
            name="claims_search",
            parameters={
                "query": query,
                "languageCode": language_code,
                "reviewPublisherSiteFilter": review_publisher_site_filter,
                "maxAgeDays": max_age_days,
                "pageSize": page_size,
                "pageToken": page_token
            }
        )

        # Create a prompt template for describing the task to the model
        prompt_template = BasePromptTemplate(f"Search for fact-checked claims using Google Fact Check API with parameters: {json.dumps(function_schema.parameters)}")

        # Create a chain to execute the function and get structured output
        chain = create_openai_function_chain(
            llm=self.llm,
            function_schema=function_schema,
            prompt_template=prompt_template,
            output_parser=StrOutputParser()  # Assuming the output needs to be parsed as a string; adjust as necessary
        )

        # Invoke the chain with the parameters and return the result
        result = await chain.run()
        return result

# Example usage
if __name__ == "__main__":
    openai_api_key = "your_openai_api_key_here"
    factcheck_tool = FactCheckAPITool(openai_api_key=openai_api_key)

    async def run():
        results = await factcheck_tool.search_claims(
            query="global warming",
            review_publisher_site_filter="nytimes.com"
        )
        print(results)

    asyncio.run(run())

#
# tools\image_formatting.py
#
# This script should be a async luigi task that checks the input image current format

# And converts it to jpeg

#
# tools\newsapi.py
#
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
import requests

# Define input schema for NewsAPI
class NewsAPIInput(BaseModel):
    q: str = Field(description="Keywords or phrases to search for")
    from_date: str = Field(None, description="The oldest date for articles")
    to_date: str = Field(None, description="The newest date for articles")
    language: str = Field(None, description="Language code for the articles")
    sort_by: str = Field(None, description="Order to sort the articles")
    page_size: int = Field(None, description="Number of results to return per page")
    page: int = Field(None, description="Page number to paginate through results")

# Custom tool for NewsAPI
class NewsAPITool(BaseTool):
    name = "newsapi_tool"
    description = "Fetches articles from NewsAPI based on search criteria"
    args_schema: BaseModel = NewsAPIInput

    def _run(
        self, 
        args: NewsAPIInput, 
        run_manager=None
    ) -> dict:
        """Fetch articles from NewsAPI"""
        response = requests.get(
            'https://newsapi.org/v2/everything',
            params=args.dict(exclude_none=True),
            headers={'Authorization': 'Bearer ' + 'YOUR_API_KEY'}  # Replace with your actual API key
        )
        return response.json()

# Tool registration
newsapi_tool = NewsAPITool()

#
# tools\serpapi.py
#


#
# tools\__init__.py
#


#
# utils\check_existing_fact.py
#


#
# utils\save_new_fact.py
#


