import os
import torch
import time
import threading
from openai import OpenAI
from elevenlabs import generate, stream, set_api_key, save, play
from itertools import takewhile, tee
from diffusers import DiffusionPipeline
from dotenv import load_dotenv
from typing import Iterator
from PIL import Image
from io import BytesIO
import base64
import requests

# load env vars
load_dotenv()

ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
BASETEN_API_KEY = os.getenv('BASETEN_API_KEY')

client = OpenAI(api_key=OPENAI_API_KEY)
set_api_key(ELEVENLABS_API_KEY)

# Set the device to CPU
device = torch.device("cpu")

def generate_image_SDXL_Local(prompt, i, num_inference_steps=1, guidance_scale=0.0):

    # define pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        # torch_dtype=torch.float16,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate an image using the pipeline
    with torch.no_grad():
        # The generate function might vary based on the pipeline you are using
        results = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    # Save the generated image to a file
    image = results.images[0]
    filename = f"./static/images/generated_image_{i}.png"
    image.save(filename)

def generate_image_SDXL_API(prompt, i, num_inference_steps=1, guidance_scale=0.0):

    model_id = "5womy77q"
    BASE64_PREAMBLE = "data:image/png;base64,"

    def b64_to_pil(b64_str):
        return Image.open(BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, ""))))

    data = {
        "prompt": prompt,
        "num_steps": 4, #num_inference_steps,
    }

    # Start timing
    start_time = time.time()

    # Call model endpoint
    res = requests.post(
        f"https://model-{model_id}.api.baseten.co/production/predict",
        headers={"Authorization": f"Api-Key {BASETEN_API_KEY}"},
        json=data
    )

    # End timing
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = round(end_time - start_time, 3)
    print(f"SDXL API Request took took {elapsed_time} seconds to execute.")

    # Get output image
    res = res.json()
    img_b64 = res.get("result")

    filename = f"./static/images/generated_image_{i}.png"

    # Save the base64 string to a PNG
    image = b64_to_pil(img_b64)
    # img.show()
    image.save(filename)

def generate_image_SDXL(prompt, i, num_inference_steps=1, guidance_scale=0.0, use_local=True):

    if use_local == True:
        generate_image_SDXL_Local(
            prompt,
            i,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    else:
        generate_image_SDXL_API(
            prompt,
            i,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )



def get_image_prompt_from_gpt(paragraph, max_tokens=80, temperature=0.7):

    # set system and context prompt
    system_prompt = """Generate a single-string image prompt based on a given paragraph, delimited by double angle brackets (<<, >>)!
        You should prioritize words in the paragraph which can be depicted, such as objects and a scene.
        The output should be a continuous string suitable for stable diffusion,
        without any newlines, section headers, or colons."""
    
    context_prompt = f"""Here is a paragraph, delimited by angle brackets:
    Paragraph: << {paragraph} >>.
    Generate an Image Prompt suitable for Stable Diffusion!
    The Image Prompt should always aim to depict a scene with discerinble objects from the paragraph, as in a visual audiobook!
    The Image Prompt must be a single, continuous string without any newlines, 
    section headers, colons, or other complex structures.
    The Image Prompt must always start with "A scenic photograph where ", and the prompt logically adjusted to start with the depiciton!.
    The Image Prompt must always end with ". 3 point lighting, flash with softbox, 80mm, Canon EOS R3, f2.8, hasselblad, golden hour"!
    Keep your response to a length of {max_tokens} tokens at most!!"""

    # define a fallback image prompt in case of failed generation
    fallback_prompt = "A scenic view of a mountain at sunset"

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",  # Specify the GPT-4 model # gpt-4-1106-preview
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_prompt}
        ],
        temperature=temperature,  # Adjust for creativity level; lower is more deterministic
        max_tokens=max_tokens  # Adjust based on the expected length of the image prompt
    )

    # Extracting the prompt from the response
    if response.choices and len(response.choices) > 0:
        image_prompt = response.choices[0].message.content.strip()
    else:
        image_prompt = fallback_prompt

    print("Image prompt is:", image_prompt)

    return image_prompt

def process_paragraph(paragraph, index):
    # Get an image prompt from GPT based on the paragraph
    image_prompt = get_image_prompt_from_gpt(paragraph)
    
    # Pass the image prompt to your image generation function
    generate_image_SDXL(image_prompt, index)

def next_paragraph(f):
    return ''.join(takewhile(lambda x: x.strip(), f)).strip()

# def save_audio_stream_to_file(audio_stream: Iterator[bytes], output_file_path: str) -> None:
#     with open(output_file_path, 'wb') as output_file:
#         for chunk in audio_stream:
#             if chunk is not None:
#                 output_file.write(chunk)

#     print(f"Audio saved to {output_file_path}")

def save_audio_to_file(audio_stream, filename):
    with open(filename, 'wb') as file:
        file.write(audio_stream)

def say(s, voice_id, i):
    # text = f"What shall we debate?"
    audio_stream = generate(
        text=s,
        model="eleven_turbo_v2",
        voice=voice_id,
        stream=False
    )

    print(type(audio_stream))
    filename = f"./static/speech/generated_speech_{i}.wav"
    save_audio_to_file(audio_stream, filename)
    play(audio_stream)

def gen_speech_and_save(s, voice_id, i):
    # text = f"What shall we debate?"
    audio_stream = generate(
        text=s,
        model="eleven_turbo_v2",
        voice=voice_id,
        stream=False
    )

    filename = f"./static/speech/generated_speech_{i}.wav"
    save_audio_to_file(audio_stream, filename)
    # play(audio_stream)

def say_in_thread(paragraph, voice_id, i):
    # Define a function to run the text-to-speech in a separate thread
    def run():
        say(paragraph, voice_id, i)

    thread = threading.Thread(target=run)
    thread.start()
    return thread  # Return the thread object

# def TEST_generate_image(prompt):

#     # define pipeline
#     pipeline = DiffusionPipeline.from_pretrained(
#         "CompVis/stable-diffusion-v1-4",
#         # torch_dtype=torch.float16,
#     ).to("cuda" if torch.cuda.is_available() else "cpu")

#     # Generate an image using the pipeline
#     with torch.no_grad():
#         # The generate function might vary based on the pipeline you are using
#         results = pipeline(
#             prompt=prompt,
#             num_inference_steps=5,
#             guidance_scale=0.0,
#         )

#     # Save the generated image to a file
#     image = results.images[0]
#     image.save("./images/generated_image.png")