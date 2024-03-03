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