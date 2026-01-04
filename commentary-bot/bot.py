import os
from dotenv import load_dotenv
import google.generativeai as genai
from task.generate_openning import generate_openning

SYSTEM_PROMPT = """あなたは熱狂的な日本語F1実況者です。
必ず日本語で回答してください。
ルール:
- 実況のセリフのみを出力すること
- BGM、SE、効果音、括弧内の指示は絶対に書かないこと
- 音楽や演出の説明は書かないこと"""

model = genai.GenerativeModel('gemini-2.5-flash')

def config():
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

if __name__ == "__main__":
    config()
    country = "british"
    year = 2020
    commentary = generate_openning(model, SYSTEM_PROMPT, country, year)
    print(commentary)
    
    
