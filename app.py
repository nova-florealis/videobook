from flask import Flask, render_template, jsonify, url_for
import os
from dotenv import load_dotenv
from lib.lib import *

# load env vars
load_dotenv()

# Set parameters
ELEVENLABS_VOICE_ID = os.getenv('ELEVENLABS_VOICE_ID')
TEXT_FILE = "PrincipiaDiscordia"
USE_LOCAL_SDXL = False

app = Flask(__name__)

@app.route('/')
def index():
    # Render the main page
    return render_template('index.html')

@app.route('/get_chunk/<int:chunk_id>')
def get_chunk(chunk_id):

    file_path = os.getcwd()+f'/text_content/{TEXT_FILE}.txt'
    voice_id = ELEVENLABS_VOICE_ID

    try:
        with open(file_path) as f:
            for i, paragraph in enumerate(iter(lambda: next_paragraph(f), '')):

                # This just fast-forwards to the next chunk number
                if i == chunk_id:
                    print(i, chunk_id)

                    # Read paragraph
                    gen_speech_and_save(paragraph, voice_id, chunk_id)

                    # Get an image prompt from GPT based on the paragraph
                    image_prompt = get_image_prompt_from_gpt(paragraph)
                    
                    # Pass the image prompt to your image generation function
                    generate_image_SDXL(image_prompt, chunk_id, use_local=USE_LOCAL_SDXL)

                    # TODO: use generate output for filename
                    image_url = url_for('static', filename=f'images/generated_image_{chunk_id}.png')

                    # Assuming TTS and image generation are done here, and files are saved
                    audio_url = url_for('static', filename=f'speech/generated_speech_{chunk_id}.wav')

                    # send data to client
                    return jsonify({'paragraph': paragraph, 'image_url': image_url, 'audio_url': audio_url})

        # If the chunk_id is out of range
        return jsonify({'end': True})
    
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True, threaded=True,port=9000)

# https://stackoverflow.com/questions/51079338/audio-livestreaming-with-python-flask
# https://www.baseten.co/library/sdxl-turbo/
# https://docs.bentoml.org/en/1.2/use-cases/diffusion-models/sdxl-turbo.html