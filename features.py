import pandas as pd


def get_h2h(h2h):
    h2h["game"] = h2h.apply(
        lambda r: "-".join(sorted([r.winner_name, r.loser_name])), axis=1
    )
    h2h = h2h.sort_values(["game", "tourney_date"], ignore_index=True)
    h2h["left_win"] = h2h.apply(
        lambda r: f"{r.winner_name}-{r.loser_name}" == r.game, axis=1
    )
    h2h["right_win"] = ~h2h.left_win
    h2h[["left_win_cumsum", "right_win_cumsum"]] = h2h.groupby("game")[
        ["left_win", "right_win"]
    ].cumsum()
    h2h[["winner_h2h", "loser_h2h"]] = (
        h2h.groupby("game")[["left_win_cumsum", "right_win_cumsum"]]
        .shift()
        .fillna(0)
        .astype(int)
    )
    f = (
        lambda r: [r.winner_h2h, r.loser_h2h]
        if f"{r.winner_name}-{r.loser_name}" == r.game
        else [r.loser_h2h, r.winner_h2h]
    )
    h2h[["winner_h2h", "loser_h2h"]] = h2h.apply(f, axis=1).apply(pd.Series)
    h2h = h2h.drop(
        ["game", "left_win", "right_win", "left_win_cumsum", "right_win_cumsum"], axis=1
    )
    return h2h
