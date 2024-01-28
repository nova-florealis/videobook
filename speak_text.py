import os
from dotenv import load_dotenv
from lib.lib import *

# load env vars
load_dotenv()

# Set parameters
ELEVENLABS_VOICE_ID = os.getenv('ELEVENLABS_VOICE_ID')
TEXT_FILE = "PrincipiaDiscordia"

if __name__ == "__main__":

    print("Read book...")

    # Set file path
    file_path = os.getcwd()+f'/text_content/{TEXT_FILE}.txt'

    with open(file_path) as f:
        i = 0
        paragraph = next_paragraph(f)
        while paragraph:
            print("Chunk:", i)
            # say(paragraph)
            # say_thread = say_in_thread(paragraph, ELEVENLABS_VOICE_ID, i)
            say(paragraph, ELEVENLABS_VOICE_ID, i)
            process_paragraph(paragraph, i)
            # say_thread.join()
            paragraph = next_paragraph(f)

            i += 1
