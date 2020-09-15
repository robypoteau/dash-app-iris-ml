
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

clf=RandomForestClassifier(n_estimators=100)

scores = cross_validate(clf, X_train, y_train, cv=5, scoring=['f1_macro','recall_macro', 'precision_macro'])
scores_table = {
    'F1': pd.Series([scores['test_f1_macro'].mean(), scores['test_f1_macro'].std()*2], index=['mean', 'std']),
    'Recall': pd.Series([scores['test_recall_macro'].mean(), scores['test_recall_macro'].std()*2], index=['mean', 'std']),
    'Precision': pd.Series([scores['test_precision_macro'].mean(), scores['test_precision_macro'].std()*2], index=['mean', 'std'])}
pd.DataFrame(scores_table).to_csv('../results/scores.csv')

clf.fit(X_train,y_train)
feature_imp = pd.Series(clf.feature_importances_,index=iris.feature_names).sort_values(ascending=False)
# feature_imp.columns = ['Importance']
feature_imp.to_csv('../results/feature_importances.csv')

y_pred=clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm = pd.DataFrame(cm)
cm.columns = iris.target_names
cm.index = iris.target_names
cm.to_csv('../results/confusion_matrix.csv')
