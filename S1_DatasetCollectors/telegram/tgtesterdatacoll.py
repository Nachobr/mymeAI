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