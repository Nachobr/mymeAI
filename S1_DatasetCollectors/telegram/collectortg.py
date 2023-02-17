from telethon.sync import TelegramClient
import pandas as pd
import configparser
# reading the config file
config = configparser.ConfigParser()
config.read('config.ini')

api_id = config['Telegram']['api_id']
api_hash = config['Telegram']['api_hash']
phone = config['Telegram']['phone']
username = config['Telegram']['username']

data = [] 
with TelegramClient(username, api_id, api_hash) as client:
    me = client.get_me()
    print("este seria yo ", username)
    for message in client.iter_messages("tg group invite link"):
        if message.sender_id == me.id:  # check if message is sent by logged in user
            print(message.sender_id, ':', message.text, message.date)
            data.append([message.sender_id, message.text, message.date, message.id, message.post_author, message.views, message.peer_id.channel_id ])

df = pd.DataFrame(data, columns=["message.sender_id", "message.text", "message.date", "message.id", "message.post_author", "message.views", "message.peer_id.channel_id"]) # creates a new dataframe
df.to_csv('tgyubfounders.csv', encoding='utf-8')

