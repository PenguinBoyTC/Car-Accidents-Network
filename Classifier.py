import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

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

    return train_test_split(X, Y, test_size=0.33, random_state=42)

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

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = preprocessing()

    dt_clf = decisiontree(x_train, y_train)
    dt_acc = accuracy_score(y_test, dt_clf.predict(x_test))
    print(dt_acc)

    rf_clf = randomforest(x_train, y_train)
    rf_acc = accuracy_score(y_test, rf_clf.predict(x_test))
    print(rf_acc)

    lc = logisticregression(x_train, y_train)
    lc_acc = accuracy_score(y_test, lc.predict(x_test))
    print(lc_acc)

    # sv_clf = svmclassifier(x_train, y_train)
    # sv_acc = accuracy_score(y_test, sv_clf.predict(x_test))
    # print(sv_acc)


