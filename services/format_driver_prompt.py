import json

def format_driver_prompt(session_key) -> str:
    """Format driver information into a prompt string."""
    data = f"./data/{session_key}/driver_info.json"
    with open(data, "r", encoding="utf-8") as f:
        drivers = json.load(f)

    """
    [
  {"driver_name":"Lewis Hamilton","driver_number":44,"driver_team":"Mercedes"},
  {"driver_name":"Valtteri Bottas","driver_number":77,"driver_team":"Mercedes"}...]
    """
    prompt_lines = ["ドライバー情報:"]
    for driver in drivers:
        line = f"ドライバー#{driver['driver_number']}: {driver['driver_name']} - チーム: {driver['driver_team']}"
        prompt_lines.append(line)
    return "\n".join(prompt_lines)
