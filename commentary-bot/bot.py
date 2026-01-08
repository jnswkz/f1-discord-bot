import asyncio
import aiohttp
import os
from dotenv import load_dotenv
import google.generativeai as genai
from task.generate_openning import generate_openning
from task.generate_grid_cmt import generate_grid_commentary
from task.generate_result import generate_result_commentary
from task.generate_event_cmt import generate_event_commentary

SYSTEM_PROMPT = """あなたは熱狂的な日本語F1実況者です。
必ず日本語で回答してください。

ルール:
- 実況のセリフのみを出力すること
- 擬音語・擬態語（例: ブオオオ、ギャー、キュイーン、ドドド、ザワザワ、ワー、拍手など）や背景音の描写は一切書かないこと
- 観客・無線・場内アナウンス・サウンド・雰囲気など「音」や「環境」の表現は書かないこと
- BGM、SE、効果音、括弧内の指示は絶対に書かないこと
- 音楽や演出の説明は書かないこと
- 出力はDiscordの長文制限対策として、一定量ごとに必ず「---」だけの行で区切ること
- 「---」以外の区切り記号（====、***など）は使わないこと
"""

model = genai.GenerativeModel('gemini-2.5-flash')

def config():
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_oppening_commentary(country, year):
    return generate_openning(model, SYSTEM_PROMPT, country, year)

async def get_grid_commentary(session):
    return await generate_grid_commentary(session, SYSTEM_PROMPT, model)

async def get_result_commentary(session):
    return await generate_result_commentary(session, SYSTEM_PROMPT, model)

def get_event_commentary(event_df):
    return generate_event_commentary(event_df, SYSTEM_PROMPT, model)


async def main():
    # config()
    async with aiohttp.ClientSession() as session:
        result_cmt = await get_result_commentary(session)
        print("=== Result Commentary ===")
        print(result_cmt)

if __name__ == "__main__":
    config()
    asyncio.run(main() )

    
