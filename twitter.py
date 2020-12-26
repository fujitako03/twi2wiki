import os
import logging
import time

from flask import Flask, render_template, session, \
    redirect, request, Blueprint, current_app
from flask_bootstrap import Bootstrap
import tweepy

app = Blueprint('twitter', __name__)

@app.route('/twitter_auth', methods=['GET'])
def twitter_auth():
    """ 連携アプリ認証用URLにリダイレクト """
    # tweepy でアプリのOAuth認証を行う
    auth = tweepy.OAuthHandler(current_app.config["CONSUMER_KEY"], \
        current_app.config["CONSUMER_SECRET"], current_app.config["CALLBACK_URL"])

    try:
        # 連携アプリ認証用の URL を取得
        redirect_url = auth.get_authorization_url()
        # 認証後に必要な request_token を session に保存
        session['request_token'] = auth.request_token
    except tweepy.TweepError as e:
        logging.error(str(e))

    # リダイレクト
    return redirect(redirect_url)


def get_twitter_info(n_tweet=200):
    """
    user情報を取得
    """
    # request_token と oauth_verifier のチェック
    token = session.pop('request_token', None)
    verifier = request.args.get('oauth_verifier')
    if token is None or verifier is None:
        return False  # 未認証ならFalseを返す

    # tweepy でアプリのOAuth認証を行う
    auth = tweepy.OAuthHandler(current_app.config["CONSUMER_KEY"], \
        current_app.config["CONSUMER_SECRET"], current_app.config["CALLBACK_URL"])

    # Access token, Access token secret を取得．
    auth.request_token = token
    try:
        auth.get_access_token(verifier)
    except tweepy.TweepError as e:
        logging.error(str(e))
        return {}

    # tweepy で Twitter API にアクセス
    api = tweepy.API(auth)

    # tweetを取得
    time.sleep(1)
    li_tweets = []
    # 直近n_tweet件のツイートを取得
    for tweet in tweepy.Cursor(api.user_timeline).items(n_tweet):
        # RTを除いてリストに追加
        if (not tweet.retweeted) and ('RT @' not in tweet.text):
            li_tweets.append(tweet.text)

    # ユーザ情報を取得
    user_info = api.me()

    # dictにまとめる
    twitter_info = {
        "name":user_info.name,
        "screen_name":user_info.screen_name,
        "description":user_info.description,
        "pic": str(user_info.profile_image_url_https).replace("_normal",""),
        "created_at":user_info.created_at.strftime("%Y年%m月%d日"),
        "location":user_info.location,
        "tweets":li_tweets
    }

    return twitter_info


if __name__ == "__main__":
    pass