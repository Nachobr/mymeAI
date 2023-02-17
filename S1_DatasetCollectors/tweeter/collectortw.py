import tweepy
# Twitter
def collect_tweets(api_key, api_secret_key, access_token, access_token_secret, username):
    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    tweets = api.user_timeline(screen_name=username, count=200)
    tweet_text = [tweet.text for tweet in tweets]
    return tweet_text
