from discord.ext import commands
from services.sessions import get_session_data

class SessionCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(name='sessions')
    async def fetch_sessions(self, ctx, location: str):
        """Fetch and display session data for a given location."""
        sessions = await get_session_data(location)
        if sessions:
            await ctx.send(f"Fetched {len(sessions)} sessions for {location}.")
            # Here you can format the session data for a nicer output
            for session in sessions:
                await ctx.send(f"Session ID: {session['session_id']}, Date: {session['date']}, Status: {session['status']}")
        else:
            await ctx.send("No sessions found or sessions not finished.")

def setup(bot):
    bot.add_cog(SessionCommands(bot))