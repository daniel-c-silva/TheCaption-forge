# The Caption Forge

## Overview

This project is an AI-powered meme generator that creates meme captions from any image using computer vision and natural language processing. It combines image captioning with language generation to produce short, dark-humored, Gen Z-style captions rendered directly onto the image.

The result is a meme that blends absurdity, irony, and emotionally charged themes—automatically generated and saved as an image.

## Features

- Uses BLIP (Bootstrapped Language Image Pretraining) to describe images
- Generates meme captions using OpenAI's GPT-4o model
- Wraps and renders text on the image with proper formatting
- Designed for dark, minimalist meme humor without emojis or hashtags

## How It Works

1. **Image Description**  
   The script uses a pre-trained BLIP model to generate a caption from the input image.

2. **Caption Generation**  
   The generated caption is passed to GPT-4o using a carefully engineered prompt to produce a one-line meme with a Gen Z tone.

3. **Rendering**  
   The resulting caption is wrapped and rendered onto the image using Python Imaging Library (PIL), styled in white with a black outline for readability.

## Installation

### Requirements

- Python 3.7+
- Required packages:

```bash
pip install torch transformers pillow openai
Additional Setup
An OpenAI API key is required to access GPT-4o. This must be added in the script.

Ensure that sample.jpg exists or replace it with your own image.

The script requires a valid system font path (e.g., Arial on macOS or Windows).

Usage
Clone the repository and navigate to the project directory.

Add your OpenAI API key in the script:

python
Copiar
Editar
client = OpenAI(api_key="your-api-key")
Run the script:

bash
Copiar
Editar
python meme_generator.py
The generated meme will be saved as meme.jpg.

File Structure
bash
Copiar
Editar
meme_generator.py   # Main script
sample.jpg          # Input image
meme.jpg            # Output image with caption
Disclaimer
This project generates content intended for satirical and experimental purposes. The humor style may include absurd or dark themes. Please use responsibly.

License
This project is released under the MIT License.
