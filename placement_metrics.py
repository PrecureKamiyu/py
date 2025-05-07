import pandas as pd


def calculate_access_delay(servers, block_counts):
    lat = block_counts['lat_block']
    lon = block_counts['lon_block']
    counts = block_counts['count']
    # server first is lat
    # server second is lon

    total_distance = 0
    for i in range(len(lat)):
        min_distance = float('inf')
        for server in servers:
            distance = (lat[i] - server[0]) ** 2 + (lon[i] - server[1]) ** 2
            min_distance = min(min_distance, distance)
        total_distance += min_distance * counts[i]
    average_delay = total_distance / counts.sum()
    return average_delay


def calculate_workload_balance(servers, block_counts):
    lat = block_counts['lat_block']
    lon = block_counts['lon_block']
    counts = block_counts['count']
    sum_counts = counts.sum()

    server_workloads = [0 for _ in servers]
    # for every block
    # find the closest server
    # add the count to that server
    # as the workload for the server
    for i in range(len(lat)):
        closed_server = -1
        min_distance = float('inf')
        for j, server in enumerate(servers):
            distance = (lat[i] - server[0]) ** 2 + (lon[i] - server[1]) ** 2
            closed_server = j if distance < min_distance else closed_server
            min_distance = min(min_distance, distance)
        server_workloads[closed_server] += counts[i]

    num_servers = len(server_workloads)
    server_workloads = server_workloads * num_servers / sum_counts

    mean_workload = sum(server_workloads) / num_servers
    variance_sum = sum((w - mean_workload) ** 2 for w in server_workloads)
    balance = (variance_sum / num_servers) ** 0.5
    return balance


def main():
    df = pd.read_csv('./shanghai_dataset/block_counts.csv')
    lst = \
        [[ 22.5613, 117.8482],
         [ 40.4374, 122.4555],
         [ 32.2719, 112.0363]]
    result = calculate_access_delay(lst, df)
    print(f"result is {result}")
    result = calculate_workload_balance(lst, df)
    print(f"result is {result}")


if __name__ == "__main__":
    main()
