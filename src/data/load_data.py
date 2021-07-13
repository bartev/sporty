"""Functions to load various datasets"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import lower_case_col_names, drop_suffix

repo_path = Path('~/dev/github-bv/sporty').expanduser()


def load_nba(dataset='games', lowercol=True):
    """Load one of games, games_details, teams, players or rankings files

    :param dataset: games, details, teams, players, rankings
    :returns: pandas df

    """

    switcher = {
        'games': 'games.csv',
        'details': 'games_details.csv',
        'teams': 'teams.csv',
        'players': 'players.csv',
        'rankings': 'rankings.csv',
    }

    data_dir = repo_path / 'data' / 'raw' / 'wk1-nba' / 'nba-kaggle'

    fname = switcher.get(dataset, '')
    res = pd.read_csv(data_dir / fname)

    if lowercol:
        res = lower_case_col_names(res)

    return res


def get_nba_game_team_points():
    """load nba game/team/points data"""
    res0 = (
        load_nba("games")
        .query("season == 2018")[
            [
                "game_date_est",
                "game_id",
                "season",
                "home_team_id",
                "visitor_team_id",
                "pts_home",
                "pts_away",
                "home_team_wins",
            ]
        ]
        .set_index(["game_date_est", "game_id", "season", "home_team_wins"])
    )
    res1 = (
        res0[["home_team_id", "visitor_team_id"]]
        .stack()
        .reset_index(name="team_id")
        .assign(
            wl=lambda x: np.where(
                x["level_4"] == "home_team_id",
                x["home_team_wins"],
                1 - x["home_team_wins"],
            ),
            hv=lambda x: np.where(x["level_4"] == "home_team_id", "home", "visitor"),
        )
        .drop(columns=["level_4", "home_team_wins"])
    )
    res2 = (
        res0[["pts_home", "pts_away"]]
        .stack()
        .reset_index(name="points")
        .assign(hv=lambda x: np.where(x["level_4"] == "pts_home", "home", "visitor"))
        .drop(columns=["level_4", "home_team_wins"])
    )
    return res1.merge(res2)
    # return res


def load_nba_games_dataset():
    """Process the raw kaggle data into a form more like what the class file has

    2 rows per game (1 per team)
    Get team names from the teams dataset
    Get the fg, fg3, ft, etc from the details dataset
    """

    deets = (
        load_nba('details')
        .groupby(["game_id", "team_id"])[["fgm", "fga", "fg3m", "fg3a", "ftm", "fta"]]
        .sum()
        .reset_index()
    )
    nba_teams = load_nba(dataset='teams').drop(
        columns=[
            'league_id',
            "min_year",
            'max_year',
            'abbreviation',
            'yearfounded',
            'arena',
            'arenacapacity',
            'owner',
            'generalmanager',
            'headcoach',
            'dleagueaffiliation',
        ]
    )

    game_data = load_nba('games').query("season == 2018")
    game_cols = game_data.columns

    home_cols = [col for col in game_cols if col.endswith('home')]
    away_cols = [col for col in game_cols if col.endswith('away')]
    # drop_cols = 'game_status_text'
    common_cols = ["game_date_est", "game_id", "season", "home_team_wins"]

    away_data = (
        game_data[common_cols + away_cols]
        .pipe(lambda x: drop_suffix(x, "_away"))
        .assign(ha='away')
        .assign(wl=lambda x: np.where(x['home_team_wins'] == 0, 'W', 'L'))
    )

    home_data = (
        game_data[common_cols + home_cols]
        .pipe(lambda x: drop_suffix(x, "_home"))
        .assign(ha='home')
        .assign(wl=lambda x: np.where(x['home_team_wins'] == 1, 'W', 'L'))
    )

    res = (
        pd.concat([home_data, away_data])
        .merge(nba_teams)
        .merge(deets)
        .dropna()
        .sort_values(['game_date_est', 'game_id'])
        .assign(game_id=lambda x: x['game_id'].apply(str))
        .assign(season=lambda x: x['season'].apply(int))
        .assign(home_team_wins=lambda x: x['home_team_wins'].apply(int))
        .assign(team_id=lambda x: x['team_id'].apply(str))
        .rename(columns={'game_date_est': 'game_date'})
    )

    return res
