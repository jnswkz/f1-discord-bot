from typing import List, Dict, Any
import json

async def list_sessions(sessions: List[Dict[str, Any]]) -> str:
    if not sessions:
        return "No sessions available."
    
    session_list = []
    for session in sessions:
        session_info = f"{session['session_id']} - {session['date']} {session['month']} {session['time']} ({session['status']})"
        session_list.append(session_info)
    
    return "\n".join(session_list)