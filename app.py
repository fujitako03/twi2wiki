# -*- coding: utf-8 -*-
import os
import logging

from flask import Flask, render_template, url_for
from flask_bootstrap import Bootstrap

import wiki
import twitter

app = Flask(__name__)
app.secret_key = os.environ['SECRET_KEY']
app.register_blueprint(wiki.app)
app.register_blueprint(twitter.app)
bootstrap = Bootstrap(app)

# Callback URL (認証後リダイレクトされるURL)
app.config["CALLBACK_URL"] = 'https://twi2wiki.herokuapp.com/wiki'  # Heroku上
# app.config["CALLBACK_URL"] = 'http://localhost:5000/wiki' # ローカル環境

# twitter api key
app.config["CONSUMER_KEY"] = os.environ['CONSUMER_KEY']
app.config["CONSUMER_SECRET"] = os.environ['CONSUMER_SECRET']
logging.warn('app start!')

@app.route('/')
def top():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()