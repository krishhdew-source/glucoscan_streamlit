from PIL import Image
import ollama

image_path = "/Users/richaanuragini/Documents/Rishi/glucoscan_streamlit/images/DD2B6BFB-A859-4CFB-9D00-D584D5BC158C.jpg"

prompt = """
Analyze this glucometer image. Identify ONLY the primary glucose reading (the big number).
Respond with just the number (e.g., '111'), nothing else.
"""

response = ollama.chat(
    model="granite3.2-vision:2b",#granite3.2-vision:2b,minicpm-v:8b
    messages=[
        {
            "role": "user",
            "content": prompt,
            "images": [image_path]
        }
    ]
)

print(response["message"]["content"])
