import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
import pdb
from collections import Counter

FEATURES = ['ID', 'Severity', 'Zipcode', 'Sunrise_Sunset', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
            'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition']
WEIGHTS = {
    'Severity': 1,
    'Zipcode': 0.23765594,
    'Temperature(F)': 0.1785479,
    'Pressure(in)': 0.16626287,
    'Humidity(%)': 0.16000679,
    'Wind_Speed(mph)': 0.11845901,
    'Weather_Condition': 0.06897787,
    'Sunrise_Sunset': 0.03605416,
    'Visibility(mi)':0.03403548
}
THRESHOLD = 7
FILE_NAME = 'US_Accidents_Dec21_updated.csv'
VALID_DATA = 'dataset/f10000_valid_data.csv'
EDGE_DICT = f'networks/edge_dict_{THRESHOLD}.json'
EDGE_DICT_WEIGHT = f'networks/edge_dict_weight_{THRESHOLD}.json'
# Load first 10K rows of data


def read_data(filename):
    df = pd.read_csv(filename)
    return df

# Extract 11 features out of 47 features


def extract_data(filename, output_file):
    df = pd.read_csv(filename)
    acc_df = df[FEATURES]
    acc_df = acc_df.dropna(thresh=3)
    acc_df.to_csv(output_file)


# Generate edges between any two nodes based on threshold
def convert_df_to_edge_dict(df, output_file):
    edge_dict = {
        'source': [],
        'target': [],
        'weight': []
    }
    # print(output_file)


    for index_i, row_i in df.iterrows():
        for index_j in range(index_i + 1, len(df)):
            row_j = df.iloc[index_j]
            count = 0
            if row_i['Severity'] == row_j['Severity']:
                count += 1
            if row_i['Zipcode'] == row_j['Zipcode']:
                count += WEIGHTS['Zipcode'] * 8.0
            if row_i['Sunrise_Sunset'] == row_j['Sunrise_Sunset']:
                count += WEIGHTS['Sunrise_Sunset'] * 8.0
            if abs(row_i['Temperature(F)'] - row_j['Temperature(F)']) <= 1.0:
                count += WEIGHTS['Temperature(F)'] * 8.0
            if abs(row_i['Humidity(%)'] - row_j['Humidity(%)']) <= 1.0:
                count += WEIGHTS['Humidity(%)'] * 8.0
            if abs(row_i['Pressure(in)'] - row_j['Pressure(in)']) <= 0.1:
                count += WEIGHTS['Pressure(in)'] * 8.0
            if row_i['Visibility(mi)'] == row_j['Visibility(mi)']:
                count += WEIGHTS['Visibility(mi)'] * 8.0
            if abs(row_i['Wind_Speed(mph)'] - row_j['Wind_Speed(mph)']) <= 1.0:
                count += WEIGHTS['Wind_Speed(mph)'] * 8.0
            # if abs(row_i['Precipitation(in)'] - row_j['Precipitation(in)']) <= 0.02:
            #     count += 1
            if row_i['Weather_Condition'] == row_j['Weather_Condition']:
                count += WEIGHTS['Weather_Condition'] * 8.0
            if count >= THRESHOLD:
                edge_dict['source'].append(index_i)
                edge_dict['target'].append(index_j)
                edge_dict['weight'].append(count / 9)
    with open(f'networks/{output_file}_{THRESHOLD}.json', 'w') as outfile:
        pdb.set_trace()
        json.dump(edge_dict, outfile)


# create graph from edge dict


def create_graph(edges_df):
    G = nx.from_pandas_edgelist(edges_df, edge_attr=None)
    undirect_G = G.to_undirected()
    draw_graph(undirect_G)
    nx.write_gml(undirect_G, f'networks/threshold_weight_{THRESHOLD}.gml')
    return undirect_G

# draw graph


def draw_graph(G):
    options = {
        'node_size': 1,
        'with_labels': False
    }
    nx.draw(G, **options)
    plt.show()


def read_graph_from_GML(filename):
    G = nx.read_gml(filename)
    undirect_G = G.to_undirected()
    return undirect_G


def graph_analysis(undirect_G):
    sum = 0
    print(
        f'Graph with {undirect_G.number_of_nodes()} nodes and {undirect_G.number_of_edges()} edges')
    for node, degree in nx.degree(undirect_G):
        sum += degree
    average_degree = sum / undirect_G.number_of_nodes()
    print("average degree: ", average_degree)
    average_clustering = nx.average_clustering(undirect_G)
    print("average clustering: ", average_clustering)
    global_clustering = nx.transitivity(undirect_G)
    print("global clustering: ", global_clustering)
    connected_components = sorted(
        nx.connected_components(undirect_G), key=len, reverse=True)
    print("Number of connected components: ", len(connected_components))
    max_size = 0
    # largest_component = None
    for component in nx.connected_components(undirect_G):
        if len(component) > max_size:
            max_size = len(component)
            # largest_component = component
    print("Size of largest component: ", max_size)


def CCDF(nums):
    n = np.array(nums)
    cdf = n.cumsum(0)
    l = cdf[-1]
    cdf = cdf * 1.0 / l
    ccdf = 1 - cdf
    return ccdf


def plot_CCDF(g, log_scale=False):
    degree_dict = nx.degree_histogram(g)
    np_array = np.array(degree_dict)
    ents = []
    nums = []
    for degree, count in enumerate(np_array):
        ents.append(degree)
        nums.append(count)
    ccdf = CCDF(nums)
    plt.plot(ents, ccdf)

    if log_scale:
        plt.title('CCDF of degree distribution (log scale)')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Degree\n(log scale)')
        plt.ylabel('CCDF\n(log scale)')
    else:
        plt.title('CCDF of degree distribution')
        plt.xlabel('Degree')
        plt.ylabel('CCDF')
    plt.show()


def plot_PDF(g, normalized=True, log_scale=False):
    print("Creating histogram...")
    aux_y = nx.degree_histogram(g)

    aux_x = np.arange(0, len(aux_y)).tolist()

    n_nodes = g.number_of_nodes()

    if normalized:
        for i in range(len(aux_y)):
            aux_y[i] = aux_y[i]/n_nodes

        if log_scale:
            plt.title('\nPDF of Degree Distribution (Normalized) (Log Scale)\n')
            plt.ylabel('Normalized log-log frequency')
            plt.xlabel('Degree\n(log scale)')
            plt.xscale('log')
            plt.yscale('log')
        else:
            plt.title('\nPDF: Degree Distribution (Normalized)\n')
            plt.ylabel('Normalized frequency')
            plt.xlabel('Degree')
        plt.plot(aux_x, aux_y, 'o')
        plt.show()

# acc_df = read_data(VALID_DATA)


def plot_all_features_histogram(acc_df):
    temperature_list = acc_df["Temperature(F)"].values.tolist()
    plt.hist(temperature_list, bins=150)
    plt.title(f'Histogram of Temperature(F)')
    plt.xlabel("Temperature(F)")
    plt.ylabel('Frequency')
    plt.show()

    humidity_list = acc_df["Humidity(%)"].values.tolist()
    plt.hist(humidity_list, bins=100)
    plt.title(f'Histogram of Humidity(%)')
    plt.xlabel("Humidity(%)")
    plt.ylabel('Frequency')
    plt.show()

    pressure_list = acc_df["Pressure(in)"].values.tolist()
    plt.hist(pressure_list, bins=150, range=(28.5, 31.0))
    plt.title(f'Histogram of Pressure(in)')
    plt.xlabel("Pressure(in)")
    plt.ylabel('Frequency')
    plt.show()

    visibility_list = acc_df["Visibility(mi)"].values.tolist()
    print(len(visibility_list))
    plt.hist(visibility_list, bins=50, range=(0, 15))
    plt.title(f'Histogram of Visibility(mi)')
    plt.xlabel("Visibility(mi)")
    plt.ylabel('Frequency')
    plt.show()
    # visibility_list = acc_df["Visibility(mi)"].values.tolist()
    # visibility_counts = Counter(visibility_list)
    wind_speed_list = acc_df["Wind_Speed(mph)"].values.tolist()
    print(len(wind_speed_list))
    plt.hist(wind_speed_list, bins=50)
    plt.title(f'Histogram of Wind_Speed(mph)')
    plt.xlabel("Wind_Speed(mph)")
    plt.ylabel('Frequency')
    plt.show()

    weather_condition_list = acc_df["Weather_Condition"].values.tolist()
    weather_counts = Counter(weather_condition_list)
    most_common_weather_type = [x[0] for x in weather_counts.most_common(8)]
    counts = [x[1] for x in weather_counts.most_common(8)]
    plt.bar(most_common_weather_type, counts)
    plt.title(f'Histogram of Weather_Condition')
    plt.xlabel("Weather_Condition")
    plt.ylabel('Frequency')
    plt.show()

    sunrise_sunset_list = acc_df["Sunrise_Sunset"].values.tolist()
    sunrise_sunset_counts = Counter(sunrise_sunset_list)
    most_common_sunrise_sunset_type = [x[0]
                                       for x in sunrise_sunset_counts.most_common(4)]
    counts = [x[1] for x in sunrise_sunset_counts.most_common(4)]
    plt.bar(most_common_sunrise_sunset_type, counts)
    plt.title(f'Histogram of Sunrise_Sunset')
    plt.xlabel("Sunrise_Sunset")
    plt.ylabel('Frequency')
    plt.show()

    severity_list = acc_df["Severity"].values.tolist()
    severity_counts = Counter(severity_list)
    most_common_severity_type = [x[0] for x in severity_counts.most_common(4)]
    counts = [x[1] for x in severity_counts.most_common(4)]
    plt.bar(most_common_severity_type, counts)
    plt.title(f'Histogram of Severity')
    plt.xlabel("Severity")
    plt.ylabel('Frequency')
    plt.show()

    zipcode_list = acc_df["Zipcode"].values.tolist()
    zipcode_counts = Counter(zipcode_list)
    most_common_zipcode_type = [x[0] for x in zipcode_counts.most_common(30)]
    counts = [x[1] for x in zipcode_counts.most_common(30)]
    plt.bar(most_common_zipcode_type, counts)
    plt.title(f'Histogram of Zipcode')
    plt.xlabel("Zipcode")
    plt.ylabel('Frequency')
    plt.show()

    precipitation_list = acc_df["Precipitation(in)"].values.tolist()
    plt.hist(precipitation_list, bins=50, range=(0, 0.4))
    plt.title(f'Histogram of Precipitation(in)')
    plt.xlabel("Precipitation(in)")
    plt.ylabel('Frequency')
    plt.show()

    # precipitation_coutns = Counter(precipitation_list)
    # most_common_precipitation_type = [x[0]
    #                                   for x in precipitation_coutns.most_common(5)]
    # counts = [x[1] for x in precipitation_coutns.most_common(5)]
    # plt.bar(most_common_precipitation_type, counts)
    # plt.title(f'Histogram of Precipitation(in)')
    # plt.xlabel("Precipitation(in)")
    # plt.ylabel('Frequency')
    # plt.show()


def start():
    # Load first 10K rows of data
    acc_df = read_data(VALID_DATA)
    # Extract 11 features; create edges between any two nodes based on threshold
    convert_df_to_edge_dict(acc_df, EDGE_DICT_WEIGHT)
    with open(EDGE_DICT_WEIGHT) as json_file:
        edge_dict = json.load(json_file)
        edges_df = pd.DataFrame(edge_dict)
        create_graph(edges_df)  # build graph from edge dict

    undirect_G = read_graph_from_GML(f'networks/threshold_{THRESHOLD}.gml')
    graph_analysis(undirect_G)
    undirect_G_7 = read_graph_from_GML(f'networks/threshold_{7}.gml')
    print("Threshold 7")
    graph_analysis(undirect_G_7)
    undirect_G_8 = read_graph_from_GML(f'networks/threshold_{8}.gml')
    print("Threshold 8")
    graph_analysis(undirect_G_8)

    plot_PDF(undirect_G, log_scale=False)
    plot_PDF(undirect_G, log_scale=True)
    plot_CCDF(undirect_G, log_scale=False)
    plot_CCDF(undirect_G, log_scale=True)
    plot_all_features_histogram(acc_df)


if __name__ == "__main__":
    start()
