import pickle
import pandas as pd

file_name = 'random_forest.sav'

if __name__ == '__main__':
    model = pickle.load(open('model/' + file_name, 'rb'))

    input = [['94043', 0, 60, 69, 30, 10, 10.4, 0]]

    df_input = pd.DataFrame(input,
                            columns=['Zipcode', 'Sunrise_Sunset', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
                                     'Visibility(mi)', 'Wind_Speed(mph)', 'Weather_Condition'])
    print(model.predict(df_input))