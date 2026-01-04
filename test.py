import google.generativeai as genai
import PIL.Image


# ===============================
# Load API Key
# ===============================
def load_api_key():
    try:
        with open("api_key.txt", "r") as f:
            key = f.read().strip()
            if not key:
                raise ValueError
            return key
    except Exception:
        print("❌ api_key.txt not found or empty. Please add your Google API key.")
        exit(1)


genai.configure(api_key=load_api_key())
model = genai.GenerativeModel('gemini-2.5-flash')

img = PIL.Image.open('8BFB5575-8380-4CDB-B340-6592E9A72787.jpg')
# For OCR, prompt the model to extract and format text
response = model.generate_content(["Extract all text from this image and provide coordinates.", img])
print(response.text)
