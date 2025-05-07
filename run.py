from PFL_v1 import QuadraticObjective1, QuadraticObjective2, train_pfl, plot_pareto_front_and_save
from evaluation import pareto_front_similarity
import numpy as np
import pandas as pd
import evaluation_v2
import PFL_v2_Scalable
import PFL_v2_placement
from methods_02_weighted_sum import train_weighted_sum


def one():
    custom_objectives = [
        QuadraticObjective1(),
        QuadraticObjective2()
    ]

    for i in range(10):
        pareto_points = PFL_v2_Scalable.train_scalable_pareto_front(custom_objectives)
        print(f"\nCase {i}\n")
        plot_pareto_front_and_save(pareto_points, f"fig/PFL_v2_fig_{i}")


def test():
    custom_objectives = [
        QuadraticObjective1(),
        QuadraticObjective2()
    ]

    sample_01 = train_pfl(custom_objectives)
    print(np.array(sample_01))


def generate_two():
    custom_objectives = [
        QuadraticObjective1(),
        QuadraticObjective2()
    ]

    sample_01 = train_pfl(custom_objectives)
    sample_02 = train_pfl(custom_objectives)
    pareto_points1 = np.array(sample_01)
    pareto_points2 = np.array(sample_02)
    pareto_points1 = pareto_points1[:, :2].reshape(-1, 2)
    pareto_points2 = pareto_points2[:, :2].reshape(-1, 2)

    np.save('points1.npy', pareto_points1)
    np.save('points2.npy', pareto_points2)


def evaluate(pareto_points1, pareto_points2):
    import matplotlib.pyplot as plt

    similarity = pareto_front_similarity(pareto_points1, pareto_points2)
    similarity = evaluation_v2.compare_pareto_fronts(
        pareto_points1, pareto_points2)["igd"]

    fig = plt.figure(figsize=(12, 6))  # Use the correct function call

    # Add a title for the entire figure
    fig.suptitle(
        f"Comparison of Learned Pareto Fronts\nSimilarity: {similarity:.2f}", fontsize=16, fontweight='bold')

    # First subplot
    ax1 = fig.add_subplot(1, 2, 1)  # Use add_subplot instead of plt.subplot
    ax1.scatter(pareto_points1[:, 0], pareto_points1[:,
                1], c='blue', label='Pareto Front 1')
    ax1.set_xlabel('Objective 1')
    ax1.set_ylabel('Objective 2')
    ax1.set_title('Learned Pareto Front 1')
    ax1.legend()
    ax1.grid(True)

    # Second subplot
    ax2 = fig.add_subplot(1, 2, 2)  # Use add_subplot instead of plt.subplot
    ax2.scatter(pareto_points2[:, 0], pareto_points2[:,
                1], c='red', label='Pareto Front 2')
    ax2.set_xlabel('Objective 1')
    ax2.set_ylabel('Objective 2')
    ax2.set_title('Learned Pareto Front 2')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout and display the plots
    # Adjust layout to make space for the suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def evaluate_overlapping(pareto_points1, pareto_points2):
    import matplotlib.pyplot as plt
    import numpy as np

    similarity = pareto_front_similarity(pareto_points1, pareto_points2)
    similarity = evaluation_v2.compare_pareto_fronts(
        pareto_points1, pareto_points2)["igd"]

    # Find the overlapping x-range
    x1_min, x1_max = np.min(pareto_points1[:, 0]), np.max(pareto_points1[:, 0])
    x2_min, x2_max = np.min(pareto_points2[:, 0]), np.max(pareto_points2[:, 0])

    # Determine the common range
    common_x_min = max(x1_min, x2_min)
    common_x_max = min(x1_max, x2_max)

    fig = plt.figure(figsize=(12, 6))

    fig.suptitle(f"Comparison of Learned Pareto Fronts\nSimilarity: {similarity:.2f}",
                 fontsize=16, fontweight='bold')

    # First subplot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(pareto_points1[:, 0], pareto_points1[:,
                1], c='blue', label='Pareto Front 1')
    ax1.set_xlabel('Objective 1')
    ax1.set_ylabel('Objective 2')
    ax1.set_title('Learned Pareto Front 1')
    ax1.set_xlim(common_x_min, common_x_max)  # Set common x-limits
    ax1.legend()
    ax1.grid(True)

    # Second subplot
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(pareto_points2[:, 0], pareto_points2[:,
                1], c='red', label='Pareto Front 2')
    ax2.set_xlabel('Objective 1')
    ax2.set_ylabel('Objective 2')
    ax2.set_title('Learned Pareto Front 2')
    ax2.set_xlim(common_x_min, common_x_max)  # Set common x-limits
    ax2.legend()
    ax2.grid(True)

    # Optionally, set common y-limits if desired
    y1_min, y1_max = np.min(pareto_points1[:, 1]), np.max(pareto_points1[:, 1])
    y2_min, y2_max = np.min(pareto_points2[:, 1]), np.max(pareto_points2[:, 1])
    common_y_min = max(y1_min, y2_min)
    common_y_max = min(y1_max, y2_max)
    ax1.set_ylim(common_y_min, common_y_max)
    ax2.set_ylim(common_y_min, common_y_max)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def evaluate_smooth_line(pareto_points1, pareto_points2):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp1d

    similarity = pareto_front_similarity(pareto_points1, pareto_points2)
    similarity = evaluation_v2.compare_pareto_fronts(
        pareto_points1, pareto_points2)["igd"]

    # Function to remove duplicate x-coordinates (keep point with minimum y-value)
    def remove_duplicates(points):
        # Sort by x-coordinate first
        idx = np.argsort(points[:, 0])
        x, y = points[idx, 0], points[idx, 1]
        # Use unique to get first occurrence (assuming minimization problem)
        unique_idx = np.unique(x, return_index=True)[1]
        return points[idx[unique_idx]]

    # Remove duplicates from both point sets
    pareto_points1_unique = remove_duplicates(pareto_points1)
    pareto_points2_unique = remove_duplicates(pareto_points2)

    # Find the overlapping x-range
    x1_min, x1_max = np.min(pareto_points1_unique[:, 0]), np.max(
        pareto_points1_unique[:, 0])
    x2_min, x2_max = np.min(pareto_points2_unique[:, 0]), np.max(
        pareto_points2_unique[:, 0])
    common_x_min = max(x1_min, x2_min)
    common_x_max = min(x1_max, x2_max)

    # Sort points by x-coordinate for interpolation
    idx1 = np.argsort(pareto_points1_unique[:, 0])
    idx2 = np.argsort(pareto_points2_unique[:, 0])
    x1, y1 = pareto_points1_unique[idx1, 0], pareto_points1_unique[idx1, 1]
    x2, y2 = pareto_points2_unique[idx2, 0], pareto_points2_unique[idx2, 1]

    # Create interpolation functions (cubic spline)
    f1 = interp1d(x1, y1, kind='cubic', bounds_error=False)
    f2 = interp1d(x2, y2, kind='cubic', bounds_error=False)

    # Generate smooth curves
    x_smooth = np.linspace(common_x_min, common_x_max, 100)
    y1_smooth = f1(x_smooth)
    y2_smooth = f2(x_smooth)

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(f"Comparison of Learned Pareto Fronts\nSimilarity: {similarity:.2f}",
                 fontsize=16, fontweight='bold')

    # First subplot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(
        pareto_points1_unique[:, 0], pareto_points1_unique[:, 1], c='blue', label='Points', alpha=0.5)
    ax1.plot(x_smooth, y1_smooth, 'b-', label='Pareto Front 1', linewidth=2)
    ax1.set_xlabel('Objective 1')
    ax1.set_ylabel('Objective 2')
    ax1.set_title('Learned Pareto Front 1')
    ax1.set_xlim(common_x_min, common_x_max)
    ax1.legend()
    ax1.grid(True)

    # Second subplot
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(
        pareto_points2_unique[:, 0], pareto_points2_unique[:, 1], c='red', label='Points', alpha=0.5)
    ax2.plot(x_smooth, y2_smooth, 'r-', label='Pareto Front 2', linewidth=2)
    ax2.set_xlabel('Objective 1')
    ax2.set_ylabel('Objective 2')
    ax2.set_title('Learned Pareto Front 2')
    ax2.set_xlim(common_x_min, common_x_max)
    ax2.legend()
    ax2.grid(True)

    # Optional: Set common y-limits
    y1_min, y1_max = np.min(pareto_points1_unique[:, 1]), np.max(
        pareto_points1_unique[:, 1])
    y2_min, y2_max = np.min(pareto_points2_unique[:, 1]), np.max(
        pareto_points2_unique[:, 1])
    common_y_min = max(y1_min, y2_min)
    common_y_max = min(y1_max, y2_max)
    ax1.set_ylim(common_y_min, common_y_max)
    ax2.set_ylim(common_y_min, common_y_max)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def five():
    objectives = [
        QuadraticObjective1(),
        QuadraticObjective2()
    ]

    pareto_points = train_weighted_sum(objectives)
    pareto_points = np.array(pareto_points)
    np.save('other_points.npy', pareto_points)


def seven():
    custom_objectives = [
        QuadraticObjective1(),
        QuadraticObjective2()
    ]

    sample_01 = PFL_v2_Scalable.train_scalable_pareto_front(custom_objectives)
    sample_02 = PFL_v2_Scalable.train_scalable_pareto_front(custom_objectives)
    pareto_points1 = np.array(sample_01)
    pareto_points2 = np.array(sample_02)
    pareto_points1 = pareto_points1[:, :2].reshape(-1, 2)
    pareto_points2 = pareto_points2[:, :2].reshape(-1, 2)

    np.save('points1.npy', pareto_points1)
    np.save('points2.npy', pareto_points2)


def eight():
    df = pd.read_csv('./shanghai_dataset/block_counts.csv')
    objectives = [
        PFL_v2_placement.PlacementObjective1(df, 4),
        PFL_v2_placement.PlacementObjective2(df, 4)
    ]
    result = PFL_v2_placement.train_scalable_pareto_front(objectives, 1, 4)
    df = pd.DataFrame(result)
    df.to_csv("PFL_v2_placement_result.csv")

    # (pareto points)
    points = result['pareto_points']
    PFL_v2_placement.plot_pareto_front_and_save(points, "./fig/PFL_v2_placement.png")
    np.save("points3.npy", np.array(points))


def nine():
    df = pd.read_csv('./shanghai_dataset/block_counts.csv')
    objectives = [
        PFL_v2_placement.PlacementObjective1(df, 4),
        PFL_v2_placement.PlacementObjective2(df, 4)
    ]

    for i in range(10):
        print(f"{i}-th run start")
        result = PFL_v2_placement.train_scalable_pareto_front(objectives, 1, 4)
        df = pd.DataFrame(result)
        df.to_csv(f"PFL_v2_placement_result_{i}.csv")

        # plot the pareto front
        points = df['pareto_points']
        PFL_v2_placement.plot_pareto_front_and_save(points, f"./fig/PFL_v2_placement_{i}.png")
        np.save(f"points3_{i}.npy", np.array(points))


def main():
    print("1. PFL_v2 : 10 samples saved as png")
    print("2. Eval   : eval points1.npy and points2.npy")
    print("3. PFL_v1 : test")
    print("4. PFL_v1 : 2 samples saved as points1.npy and points2.npy")
    print("5. Std    : standard sample with weighted sum, saved as other_points.npy")
    print("6. Eval   : evaluate points1 and other_points")
    print("7. PFL_v2 : 2 samples saved as points1.npy and points2.npy")
    print("8. PFL_v2 : test for placement problem")
    print("9. PFL_v2 : 10 samples for placement, very long time")

    choice = input("Please enter your choice: ")

    if choice == "1":
        one()

    elif choice == "2":
        points1 = np.load('points1.npy')
        points2 = np.load('points2.npy')
        evaluate(points1, points2)

    elif choice == "3":
        test()

    elif choice == "4":
        generate_two()

    elif choice == "5":
        five()

    elif choice == "6":
        points1 = np.load('points1.npy')
        points2 = np.load('./other_points.npy')
        evaluate_smooth_line(points1, points2)

    elif choice == "7":
        seven()

    elif choice == "8":
        eight()

    elif choice == "9":
        nine()


if __name__ == "__main__":
    main()
