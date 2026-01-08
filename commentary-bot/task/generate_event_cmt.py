import google.generativeai as genai
from services.format_driver_prompt import format_driver_prompt

def format_event_data(event_df) -> str:
    """Convert event DataFrame to readable format"""
    lines = ["イベント情報:"]
    for row in event_df.itertuples():
        lines.append(f"イベント名: {row.event_name} - 開催地: {row.location} - 日付: {row.date}")
    return "\n".join(lines)

def generate_event_commentary(event_df, SYSTEMPROMPT, model) -> str:
    formatted_event = format_event_data(event_df)
    print (formatted_event)
    
    prompt = SYSTEMPROMPT + f"""以下のイベント情報を基に、F1イベント紹介実況を生成してください。
{formatted_event}
追加ドライバー情報:
{format_driver_prompt("5768")}
スタイル:
- イベントの魅力を熱く紹介
実況:"""
    
    commentary = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            max_output_tokens=5000000,
            temperature=0.7,
        )
    )
    
    return commentary.text

