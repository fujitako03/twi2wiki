import random

import pandas as pd
from sudachipy import tokenizer
from sudachipy import dictionary

# 分かち書きトークナイザーの設定
tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.A

df_job_word = pd.read_csv("./static/model/job_similar_words.tsv", sep="\t")

random.seed(33)

def pred_job(text):
    words =  [m.surface() for m in tokenizer_obj.tokenize(text, mode) if m.part_of_speech()[0] == "名詞"]
    df_filter = df_job_word[df_job_word["word"].isin(words)]
    if len(df_filter) == 0:
        # 一つも当てはまらなかったときはランダム
        job_predict = random.choice(df_job_word["job"])
    else:
        # 最も当てはまるwordが多かった職業を選ぶ
        df_entry_jobs = df_filter.groupby("job")["word"].count().sort_values(ascending=False).reset_index()
        se_jobs_top = df_entry_jobs[df_entry_jobs["word"].rank(method='min', ascending=False) == 1]["job"]
        job_predict = random.choice(se_jobs_top)
    
    return job_predict


if __name__ == "__main__":
    text="爆笑白金台でデータをゴニョゴニョしてます 発言は〇〇の見解であり、××の△△を＊＊する☆☆☆☆☆☆☆☆☆ ダーツ"
    print(pred_job(text))