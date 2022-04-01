import glob
import os
import pandas as pd

# read all files
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from player_stats import get_h2h


import warnings
warnings.filterwarnings("ignore")


all_files = glob.glob(os.path.join('', r"./tennis_atp/*.csv"))
# This is where you adjust the scope of files
all_files = [file for file in all_files if 'atp_matches_20' in file]
df_from_each_file = (pd.read_csv(f, encoding='latin-1', low_memory=False) for f in all_files)
wta_matches = pd.concat(df_from_each_file, ignore_index=True)

# factors for predicting win/losses
rel_cols = ['tourney_name', 'surface', 'tourney_level', 'tourney_date',
            'tourney_id', 'match_num', 'winner_name', 'winner_hand', 'winner_age', 'winner_rank',
            'loser_name', 'loser_hand', 'loser_age', 'loser_rank',
            'score', 'winner_h2h', 'loser_h2h']
factors = ['name', 'hand', 'age', 'rank', 'h2h', 'win_pct']
w_factors = ['winner_' + factor for factor in factors]
l_factors = ['loser_' + factor for factor in factors]

# getting the relevant columns for model prediction
main_pred = get_h2h(wta_matches)
main_pred = main_pred[rel_cols]

# Verification
# pair = ['Su Wei Hsieh', 'Naomi Osaka']
# main_pred[main_pred['winner_name'].isin(pair) & main_pred['loser_name'].isin(pair)]

# Sorting the head to head record by chronological order
main_pred = main_pred.sort_values(['tourney_date', 'tourney_id', 'match_num']).reset_index(drop=True)

# getting the winning percentage for winners and losers
winner_df = main_pred.rename({'winner_name':'player'}, axis=1)
winner_df['win'] = 1
loser_df = main_pred.rename({'loser_name':'player'}, axis=1)
loser_df['win'] = 0
# Combine winner and loser dataframe and make sure the matches are arranged in chorological orders
match_df = pd.concat([winner_df, loser_df]).sort_values(['tourney_date', 'tourney_id', 'match_num'])

# Calculating the winning percentage for both winners and losers
match_df['win_count'] = match_df.groupby(['player'])['win'].cumsum()
match_df['match_count'] = match_df.groupby(['player']).cumcount()
match_df[['win_count']] = match_df.groupby('player')[['win_count']].shift().fillna(0).astype(int)
match_df['WinPercentage'] = round(match_df['win_count'] / match_df['match_count'] * 100, 1)
# Each record would have a winner / loser name
match_df['winner_name'] = match_df['winner_name'].fillna(match_df['player'])
match_df['loser_name'] = match_df['loser_name'].fillna(match_df['player'])

# getting the winning percentage for both winners and losers
match_df = match_df.reset_index(drop=True)
match_df.loc[match_df['player'] == match_df['loser_name'], "loser_win_pct"] = match_df['WinPercentage']
match_df.loc[match_df['player'] == match_df['winner_name'], "winner_win_pct"] = match_df['WinPercentage']

# get the dataframe for model building
match_df['loser_win_pct'] = match_df['loser_win_pct'].fillna(method='bfill', limit=1)
match_df['winner_win_pct'] = match_df['winner_win_pct'].fillna(method='ffill', limit=1)
main_pred = match_df.drop_duplicates(['tourney_date', 'tourney_id', 'match_num'], keep='last')
main_pred = main_pred.drop(['player', 'win', 'win_count', 'match_count', 'WinPercentage'], axis=1)

# model building
winner_df = main_pred[w_factors]
winner_df['win'] = 'winner'
loser_df = main_pred[l_factors]
loser_df['win'] = 'loser'

winner_df.columns = [col.replace('winner_', '') for col in winner_df.columns]
loser_df.columns = [col.replace('loser_', '') for col in loser_df.columns]
match_pred = pd.concat([winner_df, loser_df])
match_pred = match_pred.dropna()

# create model building
array = match_pred.values
X = array[:, 3:6]  # the features
Y = array[:, 6]  # the desired outcome

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)

# mnodel building plus cross validation
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('MNB', MultinomialNB()))
# models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
print('The accuracy of the following models: ')
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

model = GaussianNB()
model.fit(X_train, Y_train)

#  model testing
# rank h2h win_pct
player_1_pred = model.predict_proba([[73, 0, 67.0]])
player_1_win_prob = player_1_pred[0, 1]

player_2_pred = model.predict_proba([[92, 0, 58.9]])
player_2_win_prob = player_2_pred[0, 1]

if player_1_win_prob > player_2_win_prob:
    print(f'Player 1 chance of winning is {str(round(50+(player_1_win_prob-player_2_win_prob)/player_2_win_prob*100, 2))}%')
else:
    print(f'Player 2 chance of winning is {str(round(50+(player_2_win_prob-player_1_win_prob)/player_1_win_prob*100, 2))}%')



