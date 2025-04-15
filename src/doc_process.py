import re

with open('output.txt', 'r') as file:
    text = file.readlines()
    lines = [re.sub(r'^(Host|Guest):\s*', '', line).strip() 
             for line in text if line.strip()]

