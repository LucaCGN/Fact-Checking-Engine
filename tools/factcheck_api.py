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
