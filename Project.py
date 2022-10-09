import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np

FEATURES = ['ID', 'Severity', 'Zipcode', 'Sunrise_Sunset', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
            'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition']
THRESHOLD = 7
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


def draw_graph(G):
    options = {
        'node_size': 1,
        'with_labels': False
    }
    nx.draw(G, **options)
    plt.show()


def create_graph(edges_df):
    G = nx.from_pandas_edgelist(edges_df, edge_attr=None)
    undirect_G = G.to_undirected()
    draw_graph(undirect_G)
    # nx.write_gml(undirect_G, f'threshold_{THRESHOLD}.gml')
    return undirect_G


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


def read_graph_from_GML(filename):
    G = nx.read_gml(filename)
    undirect_G = G.to_undirected()
    return undirect_G


# acc_df = read_data(VALID_DATA)
# convert_df_to_edge_dict(acc_df, EDGE_DICT)
# with open(EDGE_DICT) as json_file:
#     edge_dict = json.load(json_file)
#     edges_df = pd.DataFrame(edge_dict)
#     create_graph(edges_df)


undirect_G = read_graph_from_GML(f'threshold_{THRESHOLD}.gml')

sum = 0
for node, degree in nx.degree(undirect_G):
    sum += degree
average_degree = sum / undirect_G.number_of_nodes()
# average_path_length = nx.average_shortest_path_length(undirect_G)
# diameter = nx.diameter(undirect_G)
average_clustering = nx.average_clustering(undirect_G)
global_clustering = nx.transitivity(undirect_G)
print("average degree: ", average_degree)
# print("average path length: ", average_path_length)
# print("diameter: ", diameter)
print("average clustering: ", average_clustering)
print("global clustering: ", global_clustering)


def CCDF(nums):
    n = np.array(nums)
    cdf = n.cumsum(0)
    l = cdf[-1]
    cdf = cdf * 1.0 / l
    ccdf = 1 - cdf
    return ccdf


def plot_CDDF(g):
    degree_dict = nx.degree_histogram(g)
    np_array = np.array(degree_dict)
    ents = []
    nums = []
    for degree, count in enumerate(np_array):
        ents.append(degree)
        nums.append(count)
    ccdf = CCDF(nums)
    plt.plot(ents, ccdf)
    plt.xlabel('Degree')
    plt.ylabel('CCDF')
    plt.xscale("log")
    plt.yscale("log")
    plt.title('CCDF of Degree Distribution')
    plt.show()


def plot_PDF(g, normalized=True):
    print("Creating histogram...")
    aux_y = nx.degree_histogram(g)

    aux_x = np.arange(0, len(aux_y)).tolist()

    n_nodes = g.number_of_nodes()

    if normalized:
        for i in range(len(aux_y)):
            aux_y[i] = aux_y[i]/n_nodes

        plt.title('\nDistribution Of Node Linkages (log-log scale)')
        plt.xlabel('Degree\n(log scale)')
        plt.ylabel('Number of Nodes\n(log scale)')
        plt.xscale("log")
        plt.yscale("log")
        plt.plot(aux_x, aux_y, 'o')
        plt.show()


plot_PDF(undirect_G)
plot_CDDF(undirect_G)
