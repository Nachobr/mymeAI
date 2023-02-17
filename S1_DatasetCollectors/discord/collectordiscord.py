import discord

# Discord
def collect_discord_messages(token, channel_id):
    client = discord.Client()
    @client.event
    async def on_ready():
        channel = client.get_channel(channel_id)
        async for message in channel.history(limit=200):
            message_text.append(message.content)
    client.run(token)
    return message_text