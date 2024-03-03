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
