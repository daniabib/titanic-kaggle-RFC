import pandas as pd

train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

description = train_set.describe()
train_set['Survived'].value_counts()
train_set['Pclass'].value_counts()
train_set['Sex'].value_counts()

# Fill missing Age with median
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

cat_features = ['Pclass', 'Sex', 'Embarked']
num_features = ["Age", "SibSp", "Parch", "Fare"]

# Inspired from stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

cat_pipe = Pipeline(steps=[
        ('imputer', MostFrequentImputer()),
         ('encoder', OneHotEncoder(sparse=False))
        ])

num_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
        ])

preprocessor = ColumnTransformer(transformers=
    [('cat_encode', OneHotEncoder(dtype='int'), cat_features),
     ('num_imputer', SimpleImputer(strategy='median'), num_features),
     ])

from sklearn.ensemble import RandomForestClassifier
pipe = Pipeline(steps=
                [('preprocessor', preprocessor),
                 ('model', RandomForestClassifier())])

X_train = train_set.drop('Survived', axis=1)
y_train = train_set['Survived']

from sklearn.model_selection import GridSearchCV, KFold
param_grid = { 
    'model__n_estimators': [200, 500],
    'model__max_features': ['auto', 'sqrt', 'log2'],
    'model__max_depth' : [4,5,6,7,8],
    'model__criterion' :['gini', 'entropy']
}
k_fold = KFold(n_splits=5, shuffle=True)
grid = GridSearchCV(pipe, param_grid=param_grid, cv=k_fold, n_jobs=-1, return_train_score=True)

grid.fit(X_train, y_train)

y_test_pred = grid.predict(test_set)

out_preds = pd.DataFrame({'PassengerId': test_set.PassengerId,
                          'Survived': y_test_pred})

out_preds.to_csv('predictions/titanic_random_forest_04.csv', index=False)
