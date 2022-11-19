import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

input_file = 'dataset/f10000_valid_data.csv'

def preprocessing():
    df = pd.read_csv(input_file)
    feature_list = ['Severity', 'Zipcode', 'Sunrise_Sunset', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Weather_Condition']
    df_sel = df[feature_list].copy()
    df_sel.dropna(subset=df_sel.columns[df_sel.isnull().mean() != 0], how='any', axis=0, inplace=True)
    target = 'Severity'

    df_sel['Zipcode'] = df_sel['Zipcode'].str[:5]

    day_night_map = {
        'Night': 0,
        'Day': 1
    }

    weather_map = {
        'Light Rain': 0,
        'Overcast': 1,
        'Mostly Cloudy': 2,
        'Snow': 3,
        'Light Snow': 4,
        'Cloudy': 5,
        'nan': 6,
        'Scattered Clouds': 7,
        'Clear': 8,
        'Partly Cloudy': 9,
        'Light Freezing Drizzle': 10,
        'Light Drizzle': 11,
        'Haze': 12,
        'Rain': 13,
        'Heavy Rain': 14,
        'Fair': 15,
        'Drizzle': 16,
        'Fog': 17,
        'Thunderstorms and Rain': 18,
        'Patches of Fog': 19,
        'Light Thunderstorms and Rain': 20,
        'Mist': 21,
        'Rain Showers': 22,
        'Light Rain Showers': 23,
        'Heavy Drizzle': 24
    }

    df_sel['Sunrise_Sunset'] = df_sel['Sunrise_Sunset'].map(day_night_map)
    df_sel['Weather_Condition'] = df_sel['Weather_Condition'].map(weather_map)

    X = df_sel.drop(target, axis=1)
    Y = df_sel[target]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    return x_train, x_test, y_train, y_test, X, Y

def decisiontree(x_train, y_train):
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 21],
        'min_samples_leaf': [1, 5, 10, 20, 50, 100]
    }

    dt = DecisionTreeClassifier()
    grid_search = GridSearchCV(dt, param_grid, cv=10, scoring='accuracy', return_train_score=True)
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_

def randomforest(x_train, y_train):
    param_grid = {
        'n_estimators': [20],
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 21],
        'min_samples_leaf': [1, 5, 10, 20, 50, 100]
    }

    rf = RandomForestClassifier()
    grid_search = GridSearchCV(rf, param_grid, cv=10, scoring='accuracy', return_train_score=True)
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_

def logisticregression(x_train, y_train):
    lr = LogisticRegression(random_state=42)
    lr.fit(x_train, y_train)
    return lr

def svmclassifier(x_train, y_train):
    sv_clf = SVC(random_state=42, kernel='linear')
    sv_clf.fit(x_train, y_train)
    return sv_clf

def gradientbooster(x_train, y_train):
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(x_train, y_train)
    return clf

def draw_importance_plot(clf):
    feature_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    sns.barplot(x=feature_imp[:10], y=feature_imp.index[:8])
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title('Visualizing Importance Features')
    # plt.legend()
    plt.tight_layout()
    plt.show(block=True)

if __name__ == '__main__':
    x_train, x_test, y_train, y_test, X, Y = preprocessing()

    acc_arr = []
    model_arr = []
    name_arr = ['Decision Tree Classifier', 'Random Forest Classifier', 'Logistic Regression Classifier', 'Gradient Booster Classifier']

    # Decision Tree
    dt_clf = decisiontree(x_train, y_train)
    dt_acc = accuracy_score(y_test, dt_clf.predict(x_test))
    acc_arr.append(dt_acc)
    model_arr.append(dt_clf)
    #
    # Random Forest
    rf_clf = randomforest(x_train, y_train)
    pickle.dump(rf_clf, open('model/random_forest.sav', 'wb'))
    rf_acc = accuracy_score(y_test, rf_clf.predict(x_test))
    acc_arr.append(rf_acc)
    model_arr.append(rf_clf)
    #
    #Logistic Regression
    lc = logisticregression(x_train, y_train)
    lc_acc = accuracy_score(y_test, lc.predict(x_test))
    acc_arr.append(lc_acc)
    model_arr.append(lc)
    #
    gb_clf = gradientbooster(x_train, y_train)
    gb_acc = accuracy_score(y_test, gb_clf.predict(x_test))
    acc_arr.append(gb_acc)
    model_arr.append(gb_clf)

    # sv_clf = svmclassifier(x_train, y_train)
    # sv_acc = accuracy_score(y_test, sv_clf.predict(x_test))
    # print(sv_acc)

    for i in range(len(acc_arr)):
        print('Model: {} Accuracy: {}'.format(name_arr[i], acc_arr[i]))

    draw_importance_plot(rf_clf)

