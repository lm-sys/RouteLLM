import json
import math
import os

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.linear_model import LogisticRegression

choices = ["A", "B", "C", "D"]
OPENAI_CLIENT = OpenAI()


def compute_tiers(model_ratings, num_tiers):
    n = len(model_ratings)
    m = num_tiers
    # pd series to list
    model_ratings_list = list(model_ratings.values)

    # init 3d np array
    dp = np.zeros((n, n, m))
    dp_split = np.zeros((n, n, m))
    for i in range(n):
        for j in range(i + 1, n):
            dp[i][j][0] = np.var(model_ratings_list[i : j + 1])

    for tier in range(1, m):
        for i in range(n):
            for j in range(i + 1, n):
                dp[i][j][tier] = 1000000000
                for l in range(i, j):
                    if dp[i][j][tier] > dp[i][l][tier - 1] + dp[l + 1][j][0]:
                        dp_split[i][j][tier] = l
                        dp[i][j][tier] = dp[i][l][tier - 1] + dp[l + 1][j][0]

    cur_n = n
    split_idx = []
    for tier in range(m - 1, 0, -1):
        split = int(dp_split[0][cur_n - 1][tier])
        split_idx.append(split)
        cur_n = split + 1

    split_idx = split_idx[::-1] + [n - 1]
    model2tier = {}
    cur_idx = 0
    for i in range(len(split_idx)):
        for j in range(cur_idx, split_idx[i] + 1):
            model_name = list(model_ratings.keys())[j]
            model2tier[model_name] = i
        cur_idx = split_idx[i] + 1
    return model2tier


def compute_elo_mle_with_tie(
    df, SCALE=400, BASE=10, INIT_RATING=1000, sample_weight=None
):
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx) // 2 :] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False, penalty=None)
    if sample_weight is not None:
        sample_weight = np.concatenate([sample_weight, sample_weight])
        lr.fit(X, Y, sample_weight=sample_weight)
    else:
        lr.fit(X, Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    # calibrate llama-2-70b-chat to 1082 if applicable
    if "llama-2-70b-chat" in models.index:
        elo_scores += 1082 - elo_scores[models["llama-2-70b-chat"]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def preprocess_battles(battles_df):
    MIN_LEN = 16

    def get_first_turn(prompt_str):
        return json.loads(prompt_str)[0].strip()

    def get_winner(row):
        if row["winner_model_a"] == 1:
            return "model_a"
        elif row["winner_model_b"] == 1:
            return "model_b"
        else:
            return "tie"

    battles_df["first_turn"] = battles_df["prompt"].apply(get_first_turn)
    battles_df["winner"] = battles_df.apply(get_winner, axis=1)
    battles_df = battles_df.loc[battles_df["first_turn"].apply(len) >= MIN_LEN]
    battles_df = battles_df[["model_a", "model_b", "winner", "first_turn"]]

    return battles_df
