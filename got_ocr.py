from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval().cpu()


# input your test image
image_file = '/Users/bahadir/Desktop/macbook_phd/hackathon/Bowers (1).jpg'

# plain texts OCR
res = model.chat(tokenizer, image_file, ocr_type='ocr')

print(res)