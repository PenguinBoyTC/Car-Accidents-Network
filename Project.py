import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import json

FEATURES = ['ID', 'Severity', 'Zipcode', 'Sunrise_Sunset', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
            'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition']
THRESHOLD = 8
FILE_NAME = 'US_Accidents_Dec21_updated.csv'
VALID_DATA = 'f10000_valid_data.csv'
EDGE_DICT = f'edge_dict_{THRESHOLD}.json'


def extract_data(filename, output_file):
    df = pd.read_csv(filename)
    acc_df = df[FEATURES]
    acc_df = acc_df.dropna(thresh=3)
    acc_df.to_csv(output_file)


def read_data(filename):
    df = pd.read_csv(filename)
    return df


def create_graph(edges_df):
    options = {
        'node_size': 1,
        'with_labels': False
    }
    G = nx.from_pandas_edgelist(edges_df, edge_attr=None)
    undirect_G = G.to_undirected()
    nx.write_gml(undirect_G, f'threshold_{THRESHOLD}.gml')
    # nx.draw(undirect_G, **options)
    # plt.show()


def convert_df_to_edge_dict(df, output_file):
    edge_dict = {
        'source': [],
        'target': []
    }
    for index_i, row_i in df.iterrows():
        for index_j in range(index_i + 1, len(df)):
            row_j = df.iloc[index_j]
            count = 0
            if row_i['Severity'] == row_j['Severity']:
                count += 1
            if row_i['Zipcode'] == row_j['Zipcode']:
                count += 1
            if row_i['Sunrise_Sunset'] == row_j['Sunrise_Sunset']:
                count += 1
            if abs(row_i['Temperature(F)'] - row_j['Temperature(F)']) <= 1.0:
                count += 1
            if abs(row_i['Humidity(%)'] - row_j['Humidity(%)']) <= 1.0:
                count += 1
            if abs(row_i['Pressure(in)'] - row_j['Pressure(in)']) <= 0.1:
                count += 1
            if row_i['Visibility(mi)'] == row_j['Visibility(mi)']:
                count += 1
            if abs(row_i['Wind_Speed(mph)'] - row_j['Wind_Speed(mph)']) <= 1.0:
                count += 1
            if abs(row_i['Precipitation(in)'] - row_j['Precipitation(in)']) <= 0.02:
                count += 1
            if row_i['Weather_Condition'] == row_j['Weather_Condition']:
                count += 1
            if count >= THRESHOLD:
                edge_dict['source'].append(index_i)
                edge_dict['target'].append(index_j)
    with open(f'{output_file}_{THRESHOLD}.json', 'w') as outfile:
        json.dump(edge_dict, outfile)


# acc_df = read_data(VALID_DATA)
# convert_df_to_edge_dict(acc_df, EDGE_DICT)
with open(EDGE_DICT) as json_file:
    edge_dict = json.load(json_file)
    edges_df = pd.DataFrame(edge_dict)
    create_graph(edges_df)


# edges_df = pd.DataFrame(
#     edge_dict
# )

# create_graph(edges_df)
# acc_df = read_data('f10000_valid_data.csv')
