import numpy as np

def main(
        path="moea_pymoo.npy",
        weight=0.5,
):
    path = path
    print("path is", path)

    front = np.load(path)
    weight = weight
    value, f1, f2 = min([(weight * f1 + (1-weight) * f2, f1, f2)  for f1, f2 in front])
    print(value, f1, f2)
    return {
        "weighted_value": value,
        "balance": f1,
        "delay":   f2,
        "f1": f1,
        "f2": f2,
    }

if __name__ == "__main__":
    main()
