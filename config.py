import glob
import os

import pandas as pd

# for dataframe checking

# pd.set_option('display.max_columns', 50)
# pd.set_option('display.width', 2000)
source_files = glob.glob(os.path.join('', r"./tennis_wta/*.csv"))
data_files = [file for file in source_files if 'wta_matches_20' in file]
df_from_each_file = (pd.read_csv(f, encoding='latin-1', low_memory=False) for f in data_files)
matches_data = pd.concat(df_from_each_file, ignore_index=True)

rel_cols = ['tourney_name', 'surface', 'tourney_level', 'tourney_date',
            'tourney_id', 'match_num', 'winner_name', 'winner_hand', 'winner_age', 'winner_rank',
            'loser_name', 'loser_hand', 'loser_age', 'loser_rank',
            'score', 'winner_h2h', 'loser_h2h']
factors = ['name', 'hand', 'age', 'rank', 'h2h', 'win_pct', 'recent_wins']
w_factors = ['winner_' + factor for factor in factors]
l_factors = ['loser_' + factor for factor in factors]


