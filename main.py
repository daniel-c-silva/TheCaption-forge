from openai import OpenAI
import torch
from PIL import Image, ImageDraw, ImageFont  # for loading images & adding text
from transformers import BlipProcessor, BlipForConditionalGeneration

# ================== SETTINGS =======================
client = OpenAI(api_key= "put ur own key"

# File paths
image_path = "sample.jpg"   # the pic we're gonna meme-ify
output_path = "meme.jpg"    # where the final meme will get saved
font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"     # font for writing text on the image

# Load BLIP processor + model from Huggingface
# Processor converts images into tensors the model understands & decodes output back to text
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# ======== Step 1: Get caption from image ===========
def get_image_caption(img_path) :
    """
    Takes an image path, opens it, and returns a short caption describing it.
    Uses BLIP model to generate caption.
    """
    # open the image and make sure it’s RGB (3 channels)
    image = Image.open(img_path).convert("RGB")

    # process image into tensors (language model input) — processor converts image to what model understands
    inputs = processor(image, return_tensors="pt")

    # no gradients here, just inference (speed + memory saving)
    with torch.no_grad():
        # generate caption tokens from image tensor inputs
        # **inputs unpacks the dict so model.generate gets named arguments
        output = model.generate(**inputs)

    # decode output tokens back to readable text, skipping special tokens (clean text)
    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption


# ====== STEP 2: Generate Meme Text with GPT ======
def get_meme_caption(image_caption):
    """
    Sends the image description to OpenAI and gets a funny Gen Z-style meme caption.
    """
    # tell ChatGPT exactly what kind of humour style to use and constraints (dark humour, absurd, no emojis)
    prompt = f"You are a gen z meme creator not the zesty meme type, the dark humoured(gambling, drinking, domestic violence, being proud of ur son for him being a delinquint and meth ,etc romanticizing it) and very ABSURD kind make it short use no emojis the only possible ponctuation is periods and try to make it a one liner with exceptions. Make a meme caption based on this description: '{image_caption}'"

    # Use the new SDK structure to call ChatGPT
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-3.5-turbo" if you're using that
        messages=[
            # system message sets ChatGPT’s overall style for this conversation
            {"role": "system", "content": "You're a chaotic Gen Z meme expert. Keep it ironic, random, and funny for our standards.."},
            {"role": "user", "content": prompt}  # user prompt with detailed instructions
        ]
    )

    # Extract and return the final meme text (the reply content)
    return response.choices[0].message.content.strip()


# ====== STEP 3: Draw Text on the Image ======   Worst part.
def draw_text_on_image(img_path, text, save_path):
    """
    Takes an image, adds meme-style caption text, and saves the result.
    
    Parameters:
    - img_path: Path to the original image file (e.g., "sample.jpg")
    - text: The meme caption text to draw on the image
    - save_path: Where to save the final meme image (e.g., "meme.jpg")
    """

    # open image for editing (ensure RGB)
    img = Image.open(img_path).convert("RGB")

    # create a drawing context so we can add text on image
    draw = ImageDraw.Draw(img)

    # load font for drawing — font file path + size in px
    font = ImageFont.truetype(font_path, size=40)

    # ----------- Step 1: Wrap text so it fits image width -----------
    lines = []                   # final list of lines to draw
    words = text.split()         # split text into individual words
    current_line = ""            # temp string to build each line

    for word in words:
        # try adding next word to current line (test if fits)
        test_line = f"{current_line} {word}".strip()

        # if this line’s pixel width > image width minus margin (40 px), wrap line
        if draw.textlength(test_line, font=font) > img.width - 40:
            lines.append(current_line)   # save current line as full
            current_line = word          # start new line with current word
        else:
            current_line = test_line     # if fits, keep adding words to current line

    # add last line after loop finishes
    lines.append(current_line)

    # ----------- Step 2: Draw each line centered with outline for readability -----------
    y = 10  # vertical start position from top of image

    for line in lines:
        # measure text bounding box to get width and height (accurate for font)
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        # draw text with white fill + black stroke outline for good visibility
        draw.text(
            ((img.width - w) / 2, y),  # center horizontally on image
            line,                      # text content to draw
            fill="white",              # white font color
            font=font,
            stroke_width=2,            # thickness of outline stroke
            stroke_fill="black"        # outline color
        )

        # move y down by height + padding (10 px) for next line
        y += h + 10

    # ----------- Step 3: Save the edited image to disk -----------
    img.save(save_path)
    print(f"[✔] Meme saved to {save_path}")


# ====== MAIN PROGRAM ======
if __name__ == "__main__":
    # Step 1: Describe the image (BLIP model outputs short descriptive caption)
    print("[1] Describing the image...")
    caption = get_image_caption(image_path)   # get caption text from image
    print("→ Description:", caption)          # print for debugging

    # Step 2: Generate meme text from description using GPT
    print("[2] Generating meme caption...")
    meme_text = get_meme_caption(caption)     # get meme-style caption text
    print("→ Meme Text:", meme_text)          # print generated meme caption

    # Step 3: Draw the meme text on image and save
    print("[3] Creating meme image...")
    draw_text_on_image(image_path, meme_text, output_path)  # add text and save final meme
    