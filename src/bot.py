from discord.ext import commands

# Initialize the bot with a command prefix
bot = commands.Bot(command_prefix='!')

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name} - {bot.user.id}')

# Load commands
bot.load_extension('commands.sessions_command')

# Run the bot with the specified token
if __name__ == "__main__":
    bot.run('YOUR_DISCORD_BOT_TOKEN')