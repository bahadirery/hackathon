from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import fitz  # PyMuPDF
from PIL import Image
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

output_image_path = "/Users/helmutbecker/Downloads/example.png"

pdf_document = pymupdf.open(img_path)

for page_num in range(len(pdf_document)):
    # Get the page
    page = pdf_document.load_page(page_num)

    # Define the resolution (higher value gives better image quality)
    zoom = 1  # 2 means 200% zoom, you can adjust it
    mat = pymupdf.Matrix(zoom, zoom)

    # Render the page as a pixmap (image object)
    pix = page.get_pixmap(matrix=mat)

    # Save the image as PNG
    output_image_path = f'output_page_{page_num + 1}.png'  # Saving each page as a separate PNG file
    pix.save(output_image_path)

    print(f'Page {page_num + 1} saved as {output_image_path}')

# Close the PDF document
pdf_document.close()

# Paths
pdf_file = "/home/baheryilmaz/projects/hackathon/helmut_ahlers.pdf"
output_folder = "/home/baheryilmaz/projects/hackathon/"

# Convert the PDF to multiple images
image_files = convert_pdf_to_images(pdf_file, output_folder)

# Load the model
model = Qwen2VLForConditionalGeneration.from_pretrained(
     "Qwen/Qwen2-VL-72B-Instruct-AWQ",
     torch_dtype=torch.float16,
     device_map="auto",
     cache_dir="/raid/work/baheryilmaz/hackathon/models",
)

# Load the processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct-AWQ")

# Initialize a list to store the output text from all pages
all_output_text = []

# Loop through each image file and process it
for image_file in image_files:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_file,  # Using the converted image
                },
                {"type": "text", "text": "Output the exact text in the image."},
            ],
        }
    ]

    # Prepare the input for the model
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    # Prepare the inputs for the model
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generate the output
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Append the output from the current page to the list
    all_output_text.append(output_text[0])

# Write all the output text to an output file
with open("output.txt", "w") as f:
    for page_num, text in enumerate(all_output_text, 1):
        f.write(f"Page {page_num}:\n")
        f.write(text)
        f.write("\n\n")