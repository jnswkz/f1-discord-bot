from docx import Document
import google.generativeai as genai

def get_document(country, year):
    country = country.lower()
    year = str(year)
    filepath = f"./commentary_bot/document/{country}_gp_{year}.docx"
    doc = Document(filepath)
    full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return full_text

def generate_openning(model, SYSTEM_PROMPT, country, year):
    document_text = get_document(country, year)
    prompt = f"""{year}年{country}グランプリのオープニング実況を生成してください。

以下の情報を基にしてください:
{document_text}

スタイル: エネルギッシュでドラマチック、テレビ中継のオープニングのように。

実況:"""
    
    commentary = model.generate_content(
        SYSTEM_PROMPT + prompt,
        generation_config=genai.GenerationConfig(
            max_output_tokens=5000000,
            temperature=0.7,
        )
    )

    return commentary.text