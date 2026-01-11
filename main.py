import discord
import dotenv
import os
from discord.ext import commands
from services.latestnews import get_latest_news
from services.sessions import get_session_data
from services.constructorStanding import get_constructor_scoreboard
from services.driverStanding import get_scoreboard
import asyncio
from typing import cast, List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import IntEnum

import pandas as pd
import aiohttp  

from services.openf1data import (
    ReplayState,
    get_car_data,
    get_laps,
    get_location,
    get_pit,
    build_pit_windows,
    build_gap_closing_windows,
    get_session_intervals,
    get_session_result,
    get_starting_grid,
    get_stints,
    get_team_radio,
    get_weather,
    run_replay,
)

from commentary_bot.bot import (
    config as configure_gemini,
    get_oppening_commentary,
    get_grid_commentary,
    get_result_commentary,
    get_event_commentary
)

# Event priority for ranking (higher = more important)
class EventPriority(IntEnum):
    OTHER = 0
    GAP_CLOSING = 1
    PIT = 2
    OVERTAKE = 3
    FINISHED = 4


@dataclass
class RaceEvent:
    """Represents a single race event for commentary."""
    timestamp: pd.Timestamp
    event_type: str
    priority: EventPriority
    description: str
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        # Higher priority first, then by timestamp
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.timestamp < other.timestamp


class ReplayManager:
    """Manages race replay state and event batching for Discord."""
    
    def __init__(self, session_key: str, channel: discord.TextChannel, batch_size: int = 5,
                 country: str = "Great Britain", year: str = "2020"):
        self.session_key = session_key
        self.channel = channel
        self.batch_size = batch_size
        self.country = country
        self.year = year
        self.events_by_t: Dict[pd.Timestamp, dict] = {}
        self.current_batch: List[RaceEvent] = []
        self.is_running = False
        self.replay_speed = 1.0  # 1.0 = realtime, 0.1 = 10x speed
        self.race_finished = False  # Track if we've seen a finish event
        
    async def _send_commentary(self, commentary: str, title: str = None):
        """Send commentary to channel, splitting by --- for Discord limits."""
        if not commentary:
            return
        
        # Split by --- delimiter
        parts = [p.strip() for p in commentary.split("---") if p.strip()]
        
        for i, part in enumerate(parts):
            if len(part) > 1900:
                # Further split if too long
                chunks = [part[j:j+1900] for j in range(0, len(part), 1900)]
                for chunk in chunks:
                    await self.channel.send(chunk)
            else:
                await self.channel.send(part)
            
            # Small delay between parts
            if i < len(parts) - 1:
                await asyncio.sleep(0.5)
    
    async def load_replay_data(self) -> bool:
        """Load all replay data from the session."""
        try:
            state = ReplayState()
            _, self.events_by_t = await run_replay(
                self.session_key,
                cache=True,
                include_overtakes=True,
                include_overtake_location=True,
                state=state,
                return_event_summary=True,
            )
            return True
        except Exception as e:
            await self.channel.send(f"Failed to load replay data: {e}")
            return False
    
    def _format_overtake(self, ov: dict) -> str:
        """Format an overtake event with location."""
        pair = ov.get("pair", (0, 0))
        a, v = pair
        zone = ov.get("zone")
        zone_type = ov.get("zone_type")
        
        if zone and zone != "Unknown":
            if zone_type == "corner":
                return f"#{a} overtakes #{v} at {zone}"
            else:
                return f"#{a} overtakes #{v} on {zone}"
        return f"#{a} overtakes #{v}"
    
    def _format_gap_closing(self, gc: dict) -> str:
        """Format a gap closing event."""
        pair = gc.get("pair", (0, 0))
        a, b = pair
        gain = gc.get("total_gain_s", 0)
        start_gap = gc.get("start_gap_s", 0)
        end_gap = gc.get("end_gap_s", 0)
        return f"#{a} closes on #{b} by {gain:.1f}s ({start_gap:.1f}s → {end_gap:.1f}s)"
    
    def _format_pit(self, driver: int) -> str:
        """Format a pit event."""
        return f"#{driver} enters the pits"
    
    def _format_finished(self, driver: int, position: Optional[int]) -> str:
        """Format a finish event."""
        if position:
            return f"#{driver} finishes P{position}"
        return f"#{driver} finishes"
    
    def _extract_events(self, t: pd.Timestamp, info: dict) -> List[RaceEvent]:
        """Extract and rank events from a timestamp's info."""
        events = []
        
        # Overtakes (highest priority for real overtakes)
        real_overtakes = info.get("real_overtakes", [])
        for ov in real_overtakes:
            events.append(RaceEvent(
                timestamp=t,
                event_type="overtake",
                priority=EventPriority.OVERTAKE,
                description=self._format_overtake(ov),
                raw_data=ov,
            ))
        
        # Pit stops
        pit_drivers = info.get("pit", [])
        for driver in pit_drivers:
            events.append(RaceEvent(
                timestamp=t,
                event_type="pit",
                priority=EventPriority.PIT,
                description=self._format_pit(driver),
                raw_data={"driver": driver},
            ))
        
        # Gap closing
        gap_closing = info.get("gap_closing", [])
        for gc in gap_closing:
            if isinstance(gc, dict):
                events.append(RaceEvent(
                    timestamp=t,
                    event_type="gap_closing",
                    priority=EventPriority.GAP_CLOSING,
                    description=self._format_gap_closing(gc),
                    raw_data=gc,
                ))
        
        # Finished
        finished = info.get("finished", [])
        for driver, position in finished:
            events.append(RaceEvent(
                timestamp=t,
                event_type="finished",
                priority=EventPriority.FINISHED,
                description=self._format_finished(driver, position),
                raw_data={"driver": driver, "position": position},
            ))
        
        return events
    
    def _format_batch_for_commentary(self, batch: List[RaceEvent]) -> str:
        """Format a batch of events for LLM commentary generation."""
        # Sort by priority (highest first)
        sorted_batch = sorted(batch)
        
        lines = []
        for event in sorted_batch:
            prefix = {
                "overtake": "[OVERTAKE]",
                "pit": "[PIT]",
                "gap_closing": "[GAP]",
                "finished": "[FINISH]",
            }.get(event.event_type, "[EVENT]")
            lines.append(f"{prefix} {event.description}")
        
        return "\n".join(lines)
    
    def _format_batch_for_llm(self, batch: List[RaceEvent]) -> str:
        """Format batch as structured prompt for LLM."""
        sorted_batch = sorted(batch)
        
        events_text = []
        for i, event in enumerate(sorted_batch, 1):
            events_text.append(f"{i}. [{event.event_type.upper()}] {event.description}")
        
        prompt = f"""Generate exciting F1 commentary for these events:

{chr(10).join(events_text)}

Keep it brief (2-3 sentences), energetic, and mention key drivers by number."""
        return prompt
    
    async def _send_batch(self, batch: List[RaceEvent]) -> bool:
        """Send a batch of events to the Discord channel with LLM commentary.
        
        Returns:
            True if a finish event was detected in this batch.
        """
        if not batch:
            return False
        
        # Check for finish events
        has_finish = any(e.event_type == "finished" for e in batch)
        
        # Sort batch by priority
        sorted_batch = sorted(batch)
        
        # Create list of description strings for LLM
        event_strings = [e.description for e in sorted_batch]
        
        # Generate LLM commentary
        try:
            commentary = get_event_commentary(event_strings)
            await self._send_commentary(commentary)
        except Exception as e:
            # Fallback: send raw events if LLM fails
            formatted = self._format_batch_for_commentary(batch)
            await self.channel.send(f"**Events:**\n{formatted}")
            print(f"Event commentary error: {e}")
        
        return has_finish
    
    async def run(self, speed: float = 1):
        """
        Run the replay second by second.
        
        Args:
            speed: Delay between seconds (0.1 = 10x speed, 1.0 = realtime)
        """
        self.is_running = True
        self.replay_speed = speed
        self.race_finished = False
        
        await self.channel.send(f"**Starting race replay for session {self.session_key}**")
        
        # Step 1: Opening commentary
        await self.channel.send("**[OPENING]**")
        try:
            opening = get_oppening_commentary(self.country, self.year)
            await self._send_commentary(opening)
        except Exception as e:
            await self.channel.send(f"Opening commentary failed: {e}")
        
        await asyncio.sleep(1)
        
        # Step 2: Grid commentary
        await self.channel.send("**[STARTING GRID]**")
        try:
            async with aiohttp.ClientSession() as session:
                grid_cmt = await get_grid_commentary(session)
                await self._send_commentary(grid_cmt)
        except Exception as e:
            await self.channel.send(f"Grid commentary failed: {e}")
        
        await asyncio.sleep(1)
        
        # Step 3: Race start message
        await self.channel.send("**[RACE START]**")
        await self.channel.send("Lights out and away we go!")
        
        await asyncio.sleep(1)
        
        # Step 4: Process events
        sorted_times = sorted(self.events_by_t.keys())
        if not sorted_times:
            await self.channel.send("No events found in replay data.")
            return
        
        start_time = sorted_times[0]
        end_time = sorted_times[-1]
        await self.channel.send(f"Replay from {start_time.strftime('%H:%M:%S')} to {end_time.strftime('%H:%M:%S')}")
        
        self.current_batch = []
        
        for t in sorted_times:
            if not self.is_running:
                await self.channel.send("Replay stopped.")
                break
            
            info = self.events_by_t[t]
            
            # Skip if no significant events
            has_events = (
                info.get("real_overtakes") or 
                info.get("pit") or 
                info.get("gap_closing") or 
                info.get("finished")
            )
            if not has_events:
                continue
            
            # Extract events from this timestamp
            events = self._extract_events(t, info)
            self.current_batch.extend(events)
            
            # Send batch if we have enough events
            if len(self.current_batch) >= self.batch_size:
                has_finish = await self._send_batch(self.current_batch)
                if has_finish:
                    self.race_finished = True
                self.current_batch = []
                await asyncio.sleep(speed)
        
        # Send remaining events
        if self.current_batch:
            has_finish = await self._send_batch(self.current_batch)
            if has_finish:
                self.race_finished = True
        
        # Step 5: Result commentary (after finish detected)
        if self.race_finished:
            await asyncio.sleep(1)
            await self.channel.send("**[RACE RESULT]**")
            try:
                async with aiohttp.ClientSession() as session:
                    result_cmt = await get_result_commentary(session)
                    await self._send_commentary(result_cmt)
            except Exception as e:
                await self.channel.send(f"Result commentary failed: {e}")
        
        self.is_running = False
        await self.channel.send("**Replay complete!**")
    
    def stop(self):
        """Stop the replay."""
        self.is_running = False


# Global replay manager storage
active_replays: Dict[int, ReplayManager] = {}  # channel_id -> ReplayManager


dotenv.load_dotenv()
TOKEN = os.getenv('TOKEN')
NEWS_CHANNEL_ID = os.getenv('NEWS_CHANNEL_ID')

# Configure Gemini API for commentary
configure_gemini()

if not TOKEN:
    raise ValueError("TOKEN environment variable is required")

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

last_news = []

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')
    client.loop.create_task(news_update())

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('$hello'):
        await message.channel.send('Hello! World!')

    if message.content.startswith('$news'):
        global last_news
        news = await get_latest_news()
        title = news[0]['title']
        href = news[0]['href']
        await message.channel.send(f'Latest News: {title}\nRead more: https://www.formula1.com{href}')     
        last_news = news
    if message.content.startswith('$sessions'):
        location = message.content.split('$sessions ')[1].strip()
        if not location:
            await message.channel.send('Please provide race location.')
        else:
            info = await get_session_data(location)
            if info:
                response = f"**F1 Sessions for {location.title()}**\n\n"
                
                for session in info:
                    session_id = session.get('session_id', 'Unknown')
                    date = session.get('date', '')
                    month = session.get('month', '')
                    time = session.get('time', '')
                    status = session.get('status', 'Unknown')
                    results = session.get('results', [])
                    
                    if status == "Not Finished":
                        status_marker = "[UPCOMING]"
                    else:
                        status_marker = "[DONE]"
                    
                    response += f"{status_marker} **{session_id}**\n"
                    response += f"{date} {month} at {time}\n"
                    response += f"Status: {status}\n"
                    
                    if results:
                        response += "**Top Results:**\n"
                        for result in results[:3]:  # Show top 3
                            pos = result.get('position', '')
                            driver = result.get('driver', '')
                            team = result.get('team', '')
                            time_or_laps = result.get('time_or_laps', result.get('time', ''))
                            response += f"  {pos}. {driver} ({team}) - {time_or_laps}\n"
                    
                    response += "\n"
                
                if len(response) > 1900:
                    chunks = []
                    current_chunk = f"**F1 Sessions for {location.title()}**\n\n"
                    
                    for session in info:
                        session_text = ""
                        session_id = session.get('session_id', 'Unknown')
                        date = session.get('date', '')
                        month = session.get('month', '')
                        time = session.get('time', '')
                        status = session.get('status', 'Unknown')
                        results = session.get('results', [])
                        
                        if status == "Not Finished":
                            status_marker = "[UPCOMING]"
                        else:
                            status_marker = "[DONE]"
                        
                        session_text += f"{status_marker} **{session_id}**\n"
                        session_text += f"{date} {month} at {time}\n"
                        session_text += f"Status: {status}\n"
                        
                        if results:
                            session_text += "**Top Results:**\n"
                            for result in results[:3]:
                                pos = result.get('position', '')
                                driver = result.get('driver', '')
                                team = result.get('team', '')
                                time_or_laps = result.get('time_or_laps', result.get('time', ''))
                                session_text += f"  {pos}. {driver} ({team}) - {time_or_laps}\n"
                        
                        session_text += "\n"
                        
                        if len(current_chunk + session_text) > 1900:
                            chunks.append(current_chunk)
                            current_chunk = session_text
                        else:
                            current_chunk += session_text
                    
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    for chunk in chunks:
                        await message.channel.send(chunk)
                else:
                    await message.channel.send(response)
            else:
                await message.channel.send(f"No session data found for '{location}'. Please check the location name.")

    if message.content.startswith('$wdc'):
        year = message.content.split('$wdc ')[1].strip()
        if not year:
            await message.channel.send('Please provide a year.')
        else:
            try:
                scoreboard = await get_scoreboard(year)
                if scoreboard:
                    response = f"World Drivers Championship {year}:\n\n"
                    for entry in scoreboard:
                        response += f"{entry['standing']}. {entry['driver']} ({entry['team']}) - {entry['points']} pts\n"
                    await message.channel.send(response)
                else:
                    await message.channel.send(f"No data found for WDC {year}.")
            except Exception as e:
                await message.channel.send(f"Error fetching WDC data: {e}")
        
    if message.content.startswith('$wcc'):
        year = message.content.split('$wcc ')[1].strip()
        if not year:
            await message.channel.send('Please provide a year.')
        else:
            try:
                scoreboard = await get_constructor_scoreboard(year)
                if scoreboard:
                    response = f"World Constructors Championship {year}:\n\n"
                    for entry in scoreboard:
                        response += f"{entry['standing']}. {entry['team']} - {entry['points']} pts\n"
                    await message.channel.send(response)
                else:
                    await message.channel.send(f"No data found for WCC {year}.")
            except Exception as e:
                await message.channel.send(f"Error fetching WCC data: {e}")
    
    # Replay commands
    if message.content.startswith('$replay'):
        parts = message.content.split()
        
        if len(parts) < 2:
            await message.channel.send(
                "**Race Replay Commands:**\n"
                "`$replay start <session_key> [country] [year]` - Start replay\n"
                "  Example: `$replay start 5768 \"Great Britain\" 2020`\n"
                "`$replay stop` - Stop current replay\n"
                "`$replay speed <0.01-1.0>` - Set replay speed (0.1 = 10x speed)\n"
                "`$replay status` - Check replay status"
            )
            return
        
        subcommand = parts[1].lower()
        channel_id = message.channel.id
        text_channel = cast(discord.TextChannel, message.channel)
        
        if subcommand == "start":
            # Check if replay is already running
            if channel_id in active_replays and active_replays[channel_id].is_running:
                await message.channel.send("A replay is already running in this channel. Use `$replay stop` first.")
                return
            
            # Parse arguments: $replay start <session_key> [country] [year] [speed]
            # Handle quoted country names
            import re
            full_content = message.content
            quoted_match = re.search(r'"([^"]+)"', full_content)
            
            session_key = parts[2] if len(parts) > 2 else SESSION_KEY
            country = "British"  # default
            year = "2020"  # default
            speed = 0.1  # default: 10x speed
            
            if quoted_match:
                country = quoted_match.group(1)
                # Find remaining args after the quoted string
                after_quote = full_content.split('"')[-1].strip().split()
                if after_quote:
                    year = after_quote[0]
                if len(after_quote) > 1:
                    try:
                        speed = float(after_quote[1])
                        speed = max(0.01, min(1.0, speed))
                    except ValueError:
                        pass
            else:
                # No quotes - simple parsing
                if len(parts) > 3:
                    country = parts[3]
                if len(parts) > 4:
                    year = parts[4]
                if len(parts) > 5:
                    try:
                        speed = float(parts[5])
                        speed = max(0.01, min(1.0, speed))
                    except ValueError:
                        pass
            
            await message.channel.send(f"Loading replay data for session {session_key} ({country} {year})...")
            
            # Create replay manager with country/year
            replay = ReplayManager(session_key, text_channel, batch_size=10, country=country, year=year)
            
            # Load data
            if await replay.load_replay_data():
                active_replays[channel_id] = replay
                # Run replay in background
                client.loop.create_task(replay.run(speed=speed))
            else:
                await message.channel.send("Failed to load replay data.")
        
        elif subcommand == "stop":
            if channel_id in active_replays:
                active_replays[channel_id].stop()
                del active_replays[channel_id]
                await message.channel.send("Replay stopped.")
            else:
                await message.channel.send("No active replay in this channel.")
        
        elif subcommand == "status":
            if channel_id in active_replays and active_replays[channel_id].is_running:
                replay = active_replays[channel_id]
                await message.channel.send(
                    f"Replay running\n"
                    f"Session: {replay.session_key}\n"
                    f"Speed: {replay.replay_speed}s delay"
                )
            else:
                await message.channel.send("No active replay in this channel.")
        
        elif subcommand == "speed":
            if len(parts) < 3:
                await message.channel.send("Usage: `$replay speed <0.01-1.0>`")
                return
            try:
                new_speed = float(parts[2])
                new_speed = max(0.01, min(1.0, new_speed))
                if channel_id in active_replays:
                    active_replays[channel_id].replay_speed = new_speed
                    await message.channel.send(f"Replay speed set to {new_speed}s delay between batches.")
                else:
                    await message.channel.send("No active replay to adjust.")
            except ValueError:
                await message.channel.send("Invalid speed value. Use a number between 0.01 and 1.0")
                

async def news_update():
    global last_news
    await client.wait_until_ready()
    
    if not NEWS_CHANNEL_ID:
        print("NEWS_CHANNEL_ID not configured, skipping news updates")
        return
        
    channel = client.get_channel(int(NEWS_CHANNEL_ID))
    if not channel:
        print(f"Channel not found: {NEWS_CHANNEL_ID}")
        return
    
    text_channel = cast(discord.TextChannel, channel)
    
    while not client.is_closed():
        try:
            news = await get_latest_news()
            if news and news != last_news:
                title = news[0]['title']
                href = news[0]['href']
                await text_channel.send(f'Latest News: {title}\nRead more: https://www.formula1.com{href}')
                last_news = news
        except Exception as e:
            print(f"Error in news update: {e}")
        await asyncio.sleep(3600)


SESSION_KEY = os.getenv("SESSION_KEY", "5768")

def fmt_drivers(nums):
    return ", ".join(f"#{n}" for n in nums)


def fmt_pairs(pairs):
    # pairs: [(a,v), ...]
    return ", ".join(f"#{a}->{v}" for a, v in pairs)


def fmt_finished(items):
    # items: [(driver, position or None), ...]
    out = []
    for dn, pos in items:
        if pos:
            out.append(f"P{pos} #{dn}")
        else:
            out.append(f"#{dn}")
    return ", ".join(out)


def fmt_closing(items):
    parts = []
    for item in items:
        if isinstance(item, dict):
            pair = item.get("pair")
            if not pair:
                continue
            a, b = pair
            gain = item.get("total_gain_s")
            start = item.get("start_gap_s")
            end = item.get("end_gap_s")
            detail = (
                f"(-{gain:.2f}s; {start:.1f}s->{end:.1f}s)"
                if gain is not None and start is not None and end is not None
                else ""
            )
            parts.append(f"#{a}->{b} {detail}".strip())
        else:
            try:
                a, b = item
                parts.append(f"#{a}->{b}")
            except Exception:
                continue
    return ", ".join(parts)


async def fetch_all_endpoints(session: aiohttp.ClientSession, session_key: str):
    coros = {
        "laps": get_laps(session, session_key, cache=True),
        "stints": get_stints(session, session_key, cache=True),
        "team_radio": get_team_radio(session, session_key, cache=True),
        "weather": get_weather(session, session_key, cache=True),
        "car_data": get_car_data(session, session_key, cache=True),
        "location": get_location(session, session_key, cache=True),
        # Use cached bookends if present; fetch if missing
        "starting_grid": get_starting_grid(session, session_key, cache=True),
        "session_result": get_session_result(session, session_key, cache=True),
    }
    results = await asyncio.gather(*coros.values())
    return {name: df for name, df in zip(coros.keys(), results)}



client.run(TOKEN)