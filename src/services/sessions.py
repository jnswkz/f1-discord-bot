# File: /simple-f1-discord-bot/src/services/sessions.py

from bs4 import BeautifulSoup
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
import json

def convert_session_id_to_url_path(session_id: str) -> str:
    """Convert session ID to URL path component."""
    if ' ' in session_id and 'Practice' not in session_id:
        return session_id.replace(' ', '-').lower()
    elif session_id == 'Sprint':
        return 'sprint-results'
    elif session_id == 'Race':
        return 'race-result'
    elif "Practice" in session_id:
        num = session_id.split(' ')[1]
        return f'practice/{num}'
    else:
        return "qualifying"

async def fetch_sessions(session: aiohttp.ClientSession, sn_id: str) -> List[Dict[str, Any]]:
    """
    Fetch Formula 1 session data for a given race location.
    
    Args:
        session: aiohttp ClientSession for making HTTP requests
        sn_id: Race location identifier (e.g., "belgium", "spain")
    
    Returns:
        List of session dictionaries containing session details and results
    """
    URL = f"https://www.formula1.com/en/racing/2025/{sn_id}"
    all_sessions = []
    alltext = []

    async with session.get(URL) as response:
        if response.status == 200:
            html_content = await response.text()
            soup = BeautifulSoup(html_content, 'html.parser')
            list_items = soup.find_all('li', class_='relative z-0 overflow-hidden flex gap-px-8 py-px-32 px-px-8 lg:px-px-16 border-t-thin border-t-surface-neutral-4')
            print(f"Found {len(list_items)} session items")
            
            for item in list_items:
                current = []
                for span in item.find_all('span'):
                    current.append(span.get_text().strip())
                print(f"Session data length: {len(current)}")
                alltext.append(current)
        else:
            print(f"Failed to fetch data from {URL}, status code: {response.status}")
            return all_sessions
    for eachlist in alltext:
        if (len(eachlist) == 18):
            session_id = eachlist[8]
            if session_id == '': 
                continue
            date = eachlist[1]
            month = eachlist[2]
            time = eachlist[9]
            status = "Ended"
            
            # Convert session_id to URL path using helper function
            sid = convert_session_id_to_url_path(session_id)
            print(f"{sid} {date} {month} {time} {status}")

            meeting_key = ""
            # Use a different variable name to avoid collision
            async with aiohttp.ClientSession() as api_session:
                country_name = sn_id[0].upper() + sn_id[1:]
                url = f"https://api.openf1.org/v1/meetings?year=2025&country_name={country_name}"
                async with api_session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        # The response is directly an array, not wrapped in 'data'
                        for meeting in data:
                            if meeting['country_name'].lower() == country_name.lower():
                                meeting_key = meeting['meeting_key']
                                break
                    else:
                        print(f"Failed to fetch meeting key, status code: {response.status}")

            # Fetch session results
            async with aiohttp.ClientSession() as results_session:
                url = f"https://www.formula1.com/en/results/2025/races/{meeting_key}/{sn_id}/{sid}"
                print(url)
                async with results_session.get(url) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        soup = BeautifulSoup(html_content, 'html.parser')
                        results_table = soup.find('table', class_='f1-table f1-table-with-data w-full')
                        session_results = []
                        
                        if results_table:
                            rows = results_table.find_all('tr')[1:]
                            n = 3  # Limit to top 3 results
                            for row in rows:
                                cols = row.find_all('td')
                                if len(cols) > 3:
                                    position = cols[0].get_text(strip=True)
                                    number = cols[1].get_text(strip=True)
                                    splist = cols[2].find_all('span')
                                    driver = ''
                                    for span in splist[3:-1]:
                                        driver += span.get_text(strip=True) + ' '
                                    driver = driver.strip()
                                    team = cols[3].get_text(strip=True)
                                    time = cols[4].get_text(strip=True)
                                    session_results.append({
                                        "position": position,
                                        "number": number,
                                        "driver": driver,
                                        "team": team,
                                        "time": time,
                                    })
                                    n -= 1
                                    if n == 0:
                                        break
                        
                        session_data = {
                            "session_id": session_id,
                            "date": date,
                            "month": month,
                            "time": time,
                            "status": status,
                            "results": session_results
                        }
                        all_sessions.append(session_data)
                    else:
                        print(f"Failed to fetch results from {url}, status code: {response.status}")
        else:
            # print(eachlist)
            all_sessions.append({
                "session_id": eachlist[7],
                "date": eachlist[1],
                "month": eachlist[2],
                "time": eachlist[9],
                "status": "Not Finished",
                "results": []
            })
            
    return all_sessions

async def get_session_data(location: str) -> Optional[List[Dict[str, Any]]]:

    return await fetch_sessions(aiohttp.ClientSession(), location)



if __name__ == "__main__":
    async def main():
        race_location = "hungary"  
        async with aiohttp.ClientSession() as session:
            print(f"Fetching session data for {race_location}...")
            sessions = await fetch_sessions(session, race_location)
            if not sessions:
                print("Sessions not finished.")
            
            output_file = 'temp.json'
            with open(output_file, 'w') as f:
                json.dump(sessions, f, indent=4)
            
            print(f"Successfully saved {len(sessions)} sessions to {output_file}")
    
    asyncio.run(main())