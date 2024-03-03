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