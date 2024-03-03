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