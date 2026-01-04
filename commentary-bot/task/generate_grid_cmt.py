import sys
import asyncio
import aiohttp

sys.path.append(".")

from services.openf1data import get_starting_grid
from services.format_driver_prompt import format_driver_prompt
from aiohttp import ClientSession
import google.generativeai as genai


def format_grid_data(grid_df) -> str:
    """Convert grid DataFrame to readable format"""
    lines = ["スターティンググリッド:"]
    for row in grid_df.itertuples():
        # check for grid_penalty
        if row.grid_penalty:
            lines.append(f"P{row.position} (ペナルティ付き): ドライバー#{row.driver_number} - 予選タイム: {row.qualifying_time} - ペナルティ: {row.grid_penalty}グリッド降格")
        else:
            lines.append(f"P{row.position}: ドライバー#{row.driver_number} - 予選タイム: {row.qualifying_time}")
    return "\n".join(lines)

async def generate_grid_commentary(session: ClientSession, SYSTEMPROMPT, model) -> str:
    grid_data = await get_starting_grid(session, session_key="5768")
    formatted_grid = format_grid_data(grid_data)
    print (formatted_grid)
    
    prompt = SYSTEMPROMPT + f"""以下のスターティンググリッドを基に、レース開始前のグリッド紹介実況を生成してください。

{formatted_grid}

追加ドライバー情報:
{format_driver_prompt("5768")}
スタイル:
- ポールポジションから順番に紹介
- 各ドライバーの位置とチームを熱く紹介
- 予選のハイライトや注目ポイントにも触れる
- テレビ中継のグリッドウォークのように

実況:"""
    
    commentary = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            max_output_tokens=5000000,
            temperature=0.7,
        )
    )
    
    return commentary.text