import pandas as pd
import numpy as np

def process_coordinates_to_blocks(csv_file_path, block_size=0.01):
    """
    Process lat/long points from CSV into blocks and count occurrences.
    Removes rows with erroneous values (negative lat/long).
    block_size: size of each grid cell in degrees (default 0.01 ~ 1.1km at equator)
    """
    # Read CSV file
    df = pd.read_csv(csv_file_path)

    # Verify required columns exist
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        raise ValueError("CSV must contain 'latitude' and 'longitude' columns")

    # Remove rows with negative or invalid values
    initial_rows = len(df)

    lat_max = 32.0
    lat_min = 30.5
    lon_max = 122.0
    lon_min = 121.0

    df = \
        df[(df['latitude'] >= lat_min) & (df['longitude'] >= lon_min) &
           (df['latitude'] <= lat_max) & (df['longitude'] <= lon_max)].dropna()

    df['latitude'] = df['latitude'] - lat_min
    df['longitude'] = df['longitude'] - lon_min
    removed_rows = initial_rows - len(df)

    if len(df) == 0:
        raise ValueError("No valid coordinates remain after filtering")

    # mul and floor

    df['lat_block'] = np.floor(df['latitude'] / block_size) * block_size
    df['lon_block'] = np.floor(df['longitude'] / block_size)  * block_size

    # Calculate the grid boundaries
    min_lat = np.floor(df['lat_block'].min())
    max_lat = np.ceil (df['lat_block'].max())
    min_lon = np.floor(df['lon_block'].min())
    max_lon = np.ceil (df['lon_block'].max())

    # Group by blocks and count points
    block_counts = df.groupby(['lat_block', 'lon_block']).size().reset_index(name='count')

    # Sort by count in descending order
    block_counts = block_counts.sort_values('count', ascending=False)

    return block_counts, removed_rows

def main():
    # Replace with your CSV file path
    file_path = './data_0601_to_0615_location.csv'

    try:
        # Process the coordinates
        result, removed = process_coordinates_to_blocks(file_path, block_size=0.001)

        # Print results
        print(f"\nRemoved {removed} rows with erroneous values")
        print("\nBlocks with point counts (sorted by count):")
        print(result)

        visualize_blocks(result)
        # Save to CSV
        result.to_csv('block_counts.csv', index=False)
        print("\nResults saved to 'block_counts.csv'")

        # Basic statistics
        print(f"\nTotal blocks: {len(result)}")
        print(f"Total points: {result['count'].sum()}")
        print(f"Max points in a block: {result['count'].max()}")


    except Exception as e:
        print(f"Error: {str(e)}")

# Optional visualization
def visualize_blocks(block_counts, path="visual_result.png"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
                         block_counts['lon_block'],
                         block_counts['lat_block'],
                         s=block_counts['count'],
                         c=5 * block_counts['count'],
                         alpha=0.5)

    plt.scatter(
        [0.4145, 0.5604],               # lon
        [0.8105, 0.7149],               # lat
        s=150,
        c=['red', 'blue'],  # Different colors for different points
    )

    plt.colorbar(scatter, label='Number of Points')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Point Distribution by Blocks')
    plt.grid(True)
    plt.savefig(path)
    plt.show()

def visualize_heatmap(block_counts):
    """
    Create a heatmap visualization of point density
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Pivot the data for heatmap
    heatmap_data = block_counts.pivot(index='lat_block',
                                    columns='lon_block',
                                    values='count').fillna(0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data,
                cmap='YlOrRd', # Yellow-Orange-Red color scheme
                annot=True,    # Show count numbers
                fmt='.0f',     # Format numbers as integers
                cbar_kws={'label': 'Number of Points'},
                linewidths=0.5)

    plt.title('Heatmap of Point Distribution by Blocks')
    plt.xlabel('Longitude Blocks')
    plt.ylabel('Latitude Blocks')

    # Invert y-axis to match typical map orientation (north up)
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
