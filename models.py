import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier

os.makedirs('models', exist_ok=True)

df = pd.read_csv('data/cs-ready.csv')

X = df.drop('bomb_planted', axis=1)
y = df['bomb_planted'].astype(int)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

preprocessor.fit(X)
with open('./models/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

pipe_lr = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000, random_state=69))])
param_lr = {'classifier__C': [0.1, 1.0, 10.0]}
grid_lr = GridSearchCV(pipe_lr, param_lr, cv=3)
grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_

pipe_gb = Pipeline([('preprocessor', preprocessor), ('classifier', GradientBoostingClassifier(random_state=69))])
param_gb = {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.01, 0.1]}
grid_gb = GridSearchCV(pipe_gb, param_gb, cv=3)
grid_gb.fit(X_train, y_train)
best_gb = grid_gb.best_estimator_

pipe_cb = Pipeline([('preprocessor', preprocessor), ('classifier', CatBoostClassifier(verbose=0, random_state=69))])
param_cb = {'classifier__depth': [4, 6], 'classifier__learning_rate': [0.01, 0.1]}
grid_cb = GridSearchCV(pipe_cb, param_cb, cv=3)
grid_cb.fit(X_train, y_train)
best_cb = grid_cb.best_estimator_

pipe_bag = Pipeline([('preprocessor', preprocessor), ('classifier', BaggingClassifier(random_state=69))])
param_bag = {'classifier__n_estimators': [10, 50]}
grid_bag = GridSearchCV(pipe_bag, param_bag, cv=3)
grid_bag.fit(X_train, y_train)
best_bag = grid_bag.best_estimator_

pipe_mlp = Pipeline([('preprocessor', preprocessor), ('classifier', MLPClassifier(max_iter=1000, random_state=69))])
param_mlp = {'classifier__hidden_layer_sizes': [(128, 64, 32), (256, 128, 64)], 'classifier__alpha': [0.0001, 0.001]}
grid_mlp = GridSearchCV(pipe_mlp, param_mlp, cv=3)
grid_mlp.fit(X_train, y_train)
best_mlp = grid_mlp.best_estimator_

estimators = [
    ('lr', best_lr),
    ('gb', best_gb)
]
pipe_stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
pipe_stack.fit(X_train, y_train)

models = {
    'logistic_regression.pkl': best_lr,
    'gradient_boosting.pkl': best_gb,
    'catboost_model.pkl': best_cb,
    'bagging_model.pkl': best_bag,
    'mlp_model.pkl': best_mlp,
    'stacking_model.pkl': pipe_stack
}

for name, model in models.items():
    with open(f'./models/{name}', 'wb') as f:
        pickle.dump(model, f)
