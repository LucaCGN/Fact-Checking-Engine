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
