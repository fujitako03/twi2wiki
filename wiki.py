import datetime

from flask import Flask, render_template, session,  redirect, request, Blueprint
from flask_bootstrap import Bootstrap
from job_predict import pred_job
import twitter
import attention

app = Blueprint('wiki', __name__)

@app.route('/wiki')
def wiki():
    # twitter データを取得
    user_info = twitter.get_twitter_info()

    if user_info:
        # 職業を予測
        job = pred_job(user_info["name"]+user_info["description"]+"".join(user_info["tweets"]))
        
        dict_info = {
            "name":user_info[ "name" ],
            "screen_name":user_info["screen_name"],
            "description": user_info["description"],
            "pic":user_info["pic"],
            "create_at": user_info["created_at"],
            "location": user_info["location"],
            "job":job,
            "predict_text":attention.predict(user_info["name"]+"は日本の"+job+"。"+user_info["description"]),
        }
    else:
        dict_info = False

    return render_template('wiki.html', user_info=dict_info)