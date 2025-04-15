from google import genai
import pdfplumber
import json
all_text = ''
prompt = """
Generate a 10-minute podcast transcript (2500+ words) between Host and Guest using STRICT format, based on the provided text:
- Alternate lines between Host and Guest
- Each line must begin with either "Host: " or "Guest: " followed by dialogue
- No markdown, only plain text with newline separators
- Maintain natural conversation flow while meeting length requirements

[User Request]
Create a podcast about 2000 words about {TOPIC} between {HOST_NAME} and {GUEST_NAME} following EXACT format above. Ensure:
1. Strict alternation between Host/Guest lines
2. No markdown symbols
3. When one finishes the part, use "\n" to split the line
4. Natural but detailed dialogue

The provided text is: 
"""
with pdfplumber.open("src\input.pdf") as pdf:

    
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            all_text += page_text + "\n"
    
    pdf.close()
    
    
APIKEY = "AIzaSyD99BYfiMvhfg-Sw7lIUAiVthFAA0Ird4Q"
client = genai.Client(api_key = APIKEY)

response = client.models.generate_content(
    model = "gemini-2.0-flash",
    contents = prompt + all_text
)

with open("output.txt", 'w') as f:
    f.write(response.text)