import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from transformers import AutoModel, AutoTokenizer

output_image_file = '/home/baheryilmaz/projects/hackathon/example_snippet.png'


#Load the tokenizer and model for OCR
tokenizer = AutoTokenizer.from_pretrained('stepfun-ai/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('stepfun-ai/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval().cuda()

# Perform OCR on the converted PNG file
res = model.chat(tokenizer, output_image_file, ocr_type='ocr')

print(res)