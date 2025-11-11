from typing import List, Dict, Any

async def fetch_session_results(meeting_key: str, sn_id: str, sid: str) -> List[Dict[str, Any]]:
    """
    Fetch session results for a given meeting key, race location, and session ID.
    
    Args:
        meeting_key: The unique key for the meeting.
        sn_id: Race location identifier (e.g., "belgium", "spain").
        sid: Session ID (e.g., "practice/1", "sprint-results").
    
    Returns:
        List of session result dictionaries containing position, driver, team, and time.
    """
    results = []
    url = f"https://www.formula1.com/en/results/2025/races/{meeting_key}/{sn_id}/{sid}"
    
    async with aiohttp.ClientSession() as results_session:
        async with results_session.get(url) as response:
            if response.status == 200:
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')
                results_table = soup.find('table', class_='f1-table f1-table-with-data w-full')
                
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
                            results.append({
                                "position": position,
                                "number": number,
                                "driver": driver,
                                "team": team,
                                "time": time,
                            })
                            n -= 1
                            if n == 0:
                                break
            else:
                print(f"Failed to fetch results from {url}, status code: {response.status}")
    
    return results