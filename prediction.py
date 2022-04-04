from model import match_pred
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# create model building
array = match_pred.values
X = array[:, 3:7]  # the features
Y = array[:, 7]  # the desired outcome

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)

# model building plus cross validation
models = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')), ('LDA', LinearDiscriminantAnalysis()),
          ('KNN', KNeighborsClassifier()), ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()),
          ('MNB', MultinomialNB())]
# models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
print('The accuracy of the following models: ')
for name, model in models:
    kfold_val = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold_val, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


model = LinearDiscriminantAnalysis()
model.fit(X_train, Y_train)

#  model testing
# rank h2h win_pct, recent_wins
player_1_pred = model.predict_proba([[2, 0, 77.2, 10]])
player_1_win_prob = player_1_pred[0, 1]
player_2_pred = model.predict_proba([[77, 1, 64.5, 8]])
player_2_win_prob = player_2_pred[0, 1]

if player_1_win_prob > player_2_win_prob:
    p1_l = round(player_1_win_prob / (player_1_win_prob + player_2_win_prob)*100, 1)
    print(f'Player 1 is more likely to win. The chance of player 1 winning is {p1_l}%')
else:
    p2_l = round(player_2_win_prob / (player_1_win_prob + player_2_win_prob)*100, 1)
    print('Player 2 is more likely to win.')