import os
from random import randint
import typing as t

from measure_total_idle_time import (
    all_idle_timestamps_key as times_key, default_folder_path as logs_folder)
from pnp_logging.track_user_activity import get_existing_data

SEC_TO_HOUR = 1/(60 * 60)
"""
Notes:
    - Goal is to find time periods of the day that are most idle
    - Ideas:
        - K-means clustering
        - Find maxima of derivative of running tally of number of ticks
        - move into 1-hour or 1/2-hour bins and simply find which bins have
            the most idle ticks
        - store all ticks into 15-minute bins and use sci-py signal find_peaks
            to find which bins are the most prominent
            - O(N): one for-loop through all ticks to put ticks into bins
                    one for-loop through all bins with sci-py find_peaks to
                        find most prominent peaks
    - Usually, someone has done this already, and in C, which is faster, so 
        implementing your own K-means is a lot slower

"""

# def read_data():
#     activity_data = get_existing_data(DEFAULT_ACTIVITY_LOG_PATH)
#     idle_data = get_existing_idle_data(DEFAULT_IDLE_TIME_LOG_PATH)
#     for window in activity_data:
#         print(window,
#               'idle: ', window[key_idle],
#               'active: ', window[key_active])
#
#     print('#######################################')
#     print(idle_data)


def assign_clusters(data_points: t.List, clusters: t.List):
    """
    :param data_points: idle_data's all_idle_timestamps_key maps to a list
    of every timestamp in which the pnp machine and its computer was deemed
    idle (not including the required 10 minute threshold window of inactivity)

    :param clusters: list containing each cluster's value(timestamp)

    :return: list of tuples(assignment, cost), i:
        timestamp, i, is mapped to one cluster, k
        cost: this mapping's cost(distance from cluster to point)
    """
    cluster_groups = [[] for k in range(len(clusters))]
    assignment_with_costs = [(0, 0) for i in range(len(data_points))]
    for i in range(len(data_points)):
        timestamp = data_points[i]
        min_distance = clusters[0]  # temporarily assign to a value
        for k in range(len(clusters)):
            distance = abs(clusters[k] - timestamp)
            if distance < min_distance:
                min_distance = distance
                assignment_with_costs[i] = (k, min_distance)
        cluster_groups[assignment_with_costs[i][0]].append(data_points[i])
    return assignment_with_costs, cluster_groups


def initialize_clusters(data_points: t.List, K: int) -> list:
    """
    Given number of clusters, K, and the number of possible indexes, return a
    list of length K containing the randomly assigned indexes that map a
    cluster location to a value.

    :param data_points: list of all timestamps
    :type: list

    :param K: number of clusters
    :type: int

    :return: list of random indexes
    :type: list
    """
    init_cluster_points = []
    for k in range(K):
        random_index = randint(0, len(data_points[times_key]) - 1)
        init_cluster_points.append(data_points[random_index])
    return init_cluster_points


def recalculate_cluster_values(groups: t.List) -> list:
    """

    :param groups: 2D list of length K clusters that contains all the data points
    assigned to each cluster

    :return: 1D list of K clusters where each value is the cluster's new
    value, defined as the mean of that cluster's assigned data points
    """
    ignored_clusters = []
    new_clusters = [0]*len(groups)
    for k in range(len(groups)):
        try:
            group_mean = sum(groups[k])/len(groups[k])
            new_clusters[k] = group_mean
        except ZeroDivisionError:
            ignored_clusters.append(k)
            continue
    for cluster_index in ignored_clusters:
        new_clusters.pop(cluster_index)
    return new_clusters


def run_k_means(data_values: t.List, clusters: t.List,
                num_iters=0, max_iters=10):
    """
    Runs for max_iters iterations to optimize costs of clusters
    :param data_values: list of all data values (ie: training examples)
    :param clusters: list of cluster values
    :param num_iters: track how many iterations have occurred
    :param max_iters: max number of iterations
    :returns:
        (assignment, cost) tuples for each i example
        list of optimized cluster values(a timestamp)
    """
    assignments_costs, cluster_groups = assign_clusters(data_values, clusters)
    new_clusters = recalculate_cluster_values(cluster_groups)
    if num_iters >= max_iters:
        assignments_costs, cluster_groups = assign_clusters(data_values,
                                                            clusters)
        return assignments_costs, new_clusters
    return run_k_means(data_values, new_clusters, num_iters+1)


def calc_cluster_cost(assignment_cost_tuples):
    """
    Calculates the total cost of each cluster, k
    :param assignment_cost_tuples: (assignment, cost) tuples of length, m
    examples

    :return: total cost of all clusters
    """
    cluster_costs = 0
    for i in range(len(assignment_cost_tuples)):  # m examples
        k, cost = assignment_cost_tuples[i]
        cluster_costs += cost
    return cluster_costs


def main():
    idle_data = []
    for file_name in os.listdir(logs_folder):
        file_path = os.path.join(logs_folder, file_name)
        data = get_existing_data(file_path)
        idle_data.append(data[times_key])
    all_clusters, all_cluster_costs = [], []
    for k in range(10):
        for num_trials in range(20):
            initial_clusters = initialize_clusters(idle_data, k)
            final_assignments_and_costs, final_clusters = (
                run_k_means(idle_data, initial_clusters))
            cost = calc_cluster_cost(final_assignments_and_costs)
            all_clusters.append(final_clusters)
            all_cluster_costs.append(cost)
    min_cost, min_cluster_index = all_cluster_costs[0], 0
    for index in range(1, len(all_clusters)):
        if all_cluster_costs[index] < min_cost:
            min_cost = all_cluster_costs[index]
            min_cluster_index = index
    print('Most Prominent Dead Zones: ', all_cluster_costs[min_cluster_index])
    print('Error of these groupings in minutes', min_cost/60)
