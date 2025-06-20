import matplotlib.pyplot as plt
import numpy as np


# categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
# group1_values = [20, 35, 30, 40] # e.g., Sales for Product A
# group2_values = [25, 32, 34, 38] # e.g., Sales for Product B


def main(
        categories=[10,20,30],
        categories_name="Number of servers",
        group1_values=[26.046,35.065,50.203],
        group2_values=[21.799,29.61,29.218],
        group1_label="cBIP-L1",
        group2_label="proposed method",
        x_label="Number of Servers",
        y_label="Delay",
):
    x = np.arange(len(categories))  # the label locations (0, 1, 2, 3...)
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, group1_values, width, label=group1_label, color='skyblue')
    rects2 = ax.bar(x + width/2, group2_values, width, label=group2_label, color='lightcoral')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Optional: Add data labels on top of the bars for clarity
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.show()

def draw_delay():
    ns = [10, 20, 30]
    delays_mip = [
        26.046,
        35.065,
        50.203,
    ]
    delays_proposed = [
        21.799,
        29.61,
        29.218,
    ]
    main(
        categories=ns,
        categories_name="Number of servers",
        group1_values=delays_mip,
        group1_label="cBIP-l1",
        group2_values=delays_proposed,
        group2_label="proposed method",
        x_label="Number of servers",
        y_label="Delay",
    )

def draw_balance():
    ns = [10, 20, 30]
    balance_proposed = [
        0.628,
        1.021,
        0.990,
    ]
    balance_mip = [
        0.849,
        1.14,
        1.370,
    ]
    main(
        categories=ns,
        categories_name="Number of servers",
        group1_values=balance_mip,
        group1_label="cBIP-l1",
        group2_values=balance_proposed,
        group2_label="proposed method",
        x_label="Number of servers",
        y_label="Balance",
    )

# draw_balance()
draw_delay()
