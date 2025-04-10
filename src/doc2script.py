from transformers import AutoModelForCausalLM, AutoTokenizer
import pdfplumber
import json
all_text = ''
with pdfplumber.open("src\input.pdf") as pdf:

    
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            all_text += page_text + "\n"
    

# Initialize model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Format input using Qwen's template
prompt = """You are a podcast script architect specializing in detailed conversational expansions. Generate JSON-formatted scripts containing: 1) 8-12 thematic segments 2) 4-6 subpoints per segment 3) Natural dialogue exchanges between HOST and EXPERT. Each speaker turn must contain: a) Topic introduction b) Example/analogy c) Counterargument d) Transition hook. Maintain 150-200 words per speaker turn."""
document = ""
messages = [{"role": "user", "content": prompt + all_text}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# Generate response
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=8000)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
