import numpy as np


def generate_to_be_chosen_locations(
        number_of_to_be_chosed_locations=20,
        number_of_set=3
):
    num_points = number_of_to_be_chosed_locations
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1

    for i in range(number_of_set):
        x_coords = np.round(np.random.uniform(x_min, x_max, num_points), 3)
        y_coords = np.round(np.random.uniform(y_min, y_max, num_points), 3)

        points = np.column_stack((x_coords, y_coords))

        path = f"locations_to_be_chosen_{i+1}.npy"
        np.save(path, points)
        print("save to", f"locations_to_be_chosen_{i+1}.npy")


if __name__ == "__main__":
    generate_to_be_chosen_locations(20)
