import slack

# Slack
def collect_slack_messages(token, channel_id):
    client = slack.WebClient(token=token)
    result = client.conversations_history(channel=channel_id)
    messages = result["messages"]
    message_text = [message["text"] for message in messages]
    return message_text