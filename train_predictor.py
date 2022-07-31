import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from download_past_data import *

key_home_goals = "FTHG"
key_away_goals = "FTAG"

key_odd_home = "BWH"
key_odd_draw = "BWD"
key_odd_away = "BWA"

columns_needed = [key_home_goals, key_away_goals, key_odd_home, key_odd_draw, key_odd_away]

year_start = 2005
year_end = 2021


def get_dataframe_for_year(season_year, division):
    season_string = season_year_to_string(season_year)
    filename = get_path_to_save(season_string, division)
    df = pd.read_csv(filename, usecols=columns_needed, header=0)
    return df


def get_dataframe_for_all_years():
    df = pd.concat([get_dataframe_for_year(year, first_league_suffix) for year in range(year_start, year_end)],
                   ignore_index=True)
    if include_second:
        df_second = pd.concat([get_dataframe_for_year(year, second_league_suffix) for year in range(year_start,
                                                                                                    year_end)],
                              ignore_index=True)
        df.append(df_second)

    return df


def result_to_string(home_goals, away_goals, truncate=3):
    home_goals = int(min(home_goals, truncate))
    away_goals = int(min(away_goals, truncate))
    return str(home_goals)+"-"+str(away_goals)


def result_string_to_goals(result_string):
    home_goals = int(result_string.split("-")[0])
    away_goals = int(result_string.split("-")[1])
    return home_goals, away_goals


def dataframe_to_X_Y(df: pd.DataFrame):
    X = df.get([key_odd_home, key_odd_draw, key_odd_away]).to_numpy()
    y = [result_to_string(line[key_home_goals], line[key_away_goals]) for _, line in df.iterrows()]
    return X, y


def preprocess_X(X):
    return np.divide(1, X)


def split_val(X, y):
    return X[:-306], y[:-306], X[-306:], y[-306:]


def single_game_metric(str_true, str_pred):
    pred_home_goals, pred_away_goals = result_string_to_goals(str_pred)
    true_home_goals, true_away_goals = result_string_to_goals(str_true)
    return kicktipp_punkteverteilung(pred_home_goals, pred_away_goals, true_home_goals, true_away_goals)


def batch_metric(str_true, str_pred):
    return 306*np.mean([single_game_metric(str_game_true, str_game_pred) for str_game_true, str_game_pred in zip(
        str_true,
                                                                                                      str_pred)])


def kicktipp_punkteverteilung(pred_home, pred_away, true_home, true_away):
    if pred_home == true_home and pred_away == true_away:
        return 4
    elif pred_home - pred_away == true_home - true_away:
        return 3
    elif np.sign(pred_home - pred_away) == np.sign(true_home - true_away):
        return 2
    else:
        return 0

def score():
    return make_scorer(batch_metric, greater_is_better=True)

def train_model(X, y):

    pipe = Pipeline([('scaler', StandardScaler()),
                     ('pca', PCA()),
                    ('svc', SVC())
                    ], verbose=0, memory=abspath('./tmp'))

    param_grid = {
        'svc__C': [1, 2, 5, 10, 20, 50, 100]
    }

    search = GridSearchCV(pipe, param_grid, cv=5, scoring=score(), n_jobs=2, verbose=3, refit=True)

    search.fit(X, y)

    print(search.best_params_)
    print(search.best_score_)

    return search


def main():
    df = get_dataframe_for_all_years()
    X, y = dataframe_to_X_Y(df)
    X = preprocess_X(X)
    return train_model(X, y)


if __name__ == '__main__':
    mdl = main()