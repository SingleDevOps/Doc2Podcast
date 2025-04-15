from transformers import AutoModelForCausalLM, AutoTokenizer
import pdfplumber
import json
all_text = ''
with pdfplumber.open("src\input.pdf") as pdf:

    
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            all_text += page_text + "\n"
    
    pdf.close()
# Initialize model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Format input using Qwen's template
prompt = """
Generate a 10-minute podcast transcript (2500+ words) between Host and Guest using STRICT format, based on the provided text:
- Alternate lines between Host and Guest
- Each line must begin with either "Host: " or "Guest: " followed by dialogue
- No markdown, only plain text with newline separators
- Maintain natural conversation flow while meeting length requirements

[User Request]
Create a 2500+ word podcast about {TOPIC} between {HOST_NAME} and {GUEST_NAME} following EXACT format above. Ensure:
1. Strict alternation between Host/Guest lines
2. Minimum 125 exchanges (250 total lines)
3. No markdown symbols
4. Natural but detailed dialogue

The provided text is: 
"""
document = ""
messages = [{"role": "user", "content": prompt + all_text}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# Generate response
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=8000, min_new_tokens=1500)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
with open("script.json","w") as f:
    json.dump(response, f)
