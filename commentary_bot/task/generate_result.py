import sys
import asyncio
import aiohttp

sys.path.append(".")

from services.openf1data import get_session_result
from services.format_driver_prompt import format_driver_prompt
from aiohttp import ClientSession
import google.generativeai as genai

def format_session_data(session_df) -> str:
    """Convert session DataFrame to readable format"""
    lines = ["セッション情報:"]
    """
      {
    "driver_number": 8,
    "position": 16,
    "status": "Finished"
  },
  {
    "driver_number": 7,
    "position": 17,
    "status": "Finished"
  },
  {
    "driver_number": 20,
    "position": null,
    "status": "Retired"
  },
  {
    "driver_number": 26,
    "position": null,
    "status": "Retired"
  },
  {
    "driver_number": 27,
    "position": null,
    "status": "Did not start"
  }
]"""

    for row in session_df.itertuples():
        if row.position:
            lines.append(f"P{row.position}: ドライバー#{row.driver_number} - ステータス: {row.status}")
        else:
            lines.append(f"ドライバー#{row.driver_number} - ステータス: {row.status}")
    return "\n".join(lines)
async def generate_result_commentary(session: ClientSession, SYSTEMPROMPT, model) -> str:

    session_data = await get_session_result(session, session_key="5768")
    formatted_session = format_session_data(session_data)
    print (formatted_session)
    
    prompt = SYSTEMPROMPT + f"""以下のレース結果を基に、レース終了後の結果発表実況を生成してください。
{formatted_session}
追加ドライバー情報:
{format_driver_prompt("5768")}
スタイル:
- 上位入賞者から順番に紹介
- 各ドライバーの位置とチームを熱く紹介
- レースのハイライトや注目ポイントにも触れる
実況:"""
    
    commentary = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            max_output_tokens=5000000,
            temperature=0.7,
        )
    )
    
    return commentary.text
