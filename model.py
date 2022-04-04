import warnings

import pandas as pd

from features import get_h2h
from config import matches_data, rel_cols, w_factors, l_factors


warnings.filterwarnings("ignore")

# getting the relevant columns for model prediction
h2h_df = get_h2h(matches_data)
h2h_df = h2h_df[rel_cols]
h2h_df = h2h_df.sort_values(['tourney_date', 'tourney_id', 'match_num']).reset_index(drop=True)

# getting the winning percentage for winners and losers
winner_df = h2h_df.rename({'winner_name': 'player'}, axis=1)
winner_df['win'] = 1
loser_df = h2h_df.rename({'loser_name': 'player'}, axis=1)
loser_df['win'] = 0

# before win rate calculation
pred_df = pd.concat([winner_df, loser_df]).sort_values(['tourney_date', 'tourney_id', 'match_num'])
pred_df['winner_name'] = pred_df['winner_name'].fillna(pred_df['player'])
pred_df['loser_name'] = pred_df['loser_name'].fillna(pred_df['player'])

# recent win rate
pred_df = pred_df.reset_index(drop=True)
pred_df['recent_wins'] = pred_df.groupby('player')['win'].apply(lambda x: x.shift().rolling(20).sum())
pred_df.loc[pred_df['player'] == pred_df['loser_name'], "loser_recent_wins"] = pred_df['recent_wins']
pred_df.loc[pred_df['player'] == pred_df['winner_name'], "winner_recent_wins"] = pred_df['recent_wins']
pred_df['loser_recent_wins'] = pred_df['loser_recent_wins'].fillna(method='bfill', limit=1)
pred_df['winner_recent_wins'] = pred_df['winner_recent_wins'].fillna(method='ffill', limit=1)

# overall win rate
pred_df['win_count'] = pred_df.groupby(['player'])['win'].cumsum()
pred_df['match_count'] = pred_df.groupby(['player']).cumcount()
pred_df[['win_count']] = pred_df.groupby('player')[['win_count']].shift().fillna(0).astype(int)
pred_df['WinPercentage'] = round(pred_df['win_count'] / pred_df['match_count'] * 100, 1)
# match_df = match_df.reset_index(drop=True)
pred_df.loc[pred_df['player'] == pred_df['loser_name'], "loser_win_pct"] = pred_df['WinPercentage']
pred_df.loc[pred_df['player'] == pred_df['winner_name'], "winner_win_pct"] = pred_df['WinPercentage']
# get the dataframe for model building
pred_df['loser_win_pct'] = pred_df['loser_win_pct'].fillna(method='bfill', limit=1)
pred_df['winner_win_pct'] = pred_df['winner_win_pct'].fillna(method='ffill', limit=1)

pred_df = pred_df.drop_duplicates(['tourney_date', 'tourney_id', 'match_num'], keep='last')
pred_df = pred_df.drop(['player', 'win', 'win_count', 'match_count', 'WinPercentage'], axis=1)

win_pred = pred_df[w_factors]
win_pred['win'] = 'winner'
lose_pred = pred_df[l_factors]
lose_pred['win'] = 'loser'

win_pred.columns = [col.replace('winner_', '') for col in win_pred.columns]
lose_pred.columns = [col.replace('loser_', '') for col in lose_pred.columns]
match_pred = pd.concat([win_pred, lose_pred])
match_pred = match_pred.dropna()