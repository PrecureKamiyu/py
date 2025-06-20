import json
import matplotlib.pyplot as plt

def test(path):
    with open(path, 'r') as f:
        data = json.load(f)

    # ns
    xs = [ns["attributes"]["coordinates"][0] for ns in data["NetworkSwitch"]]
    ys = [ns["attributes"]["coordinates"][1] for ns in data["NetworkSwitch"]]

    # bs
    bsxs = [bs["attributes"]["coordinates"][0] for bs in data["BaseStation"]]
    bsys = [bs["attributes"]["coordinates"][1] for bs in data["BaseStation"]]

    # edge server
    esxs = [es["attributes"]["coordinates"][0] for es in data["EdgeServer"]]
    esys = [es["attributes"]["coordinates"][1] for es in data["EdgeServer"]]

    plt.scatter(xs, ys,
                color='red',
                alpha=0.5)

    plt.scatter(esxs, esys,
                color='blue',
                alpha=0.5)

    # nl link id pair
    nl_pairs = [[nl["relationships"]["nodes"][0]["id"], nl["relationships"]["nodes"][1]["id"]] for nl in data["NetworkLink"]]
    for id1, id2 in nl_pairs:
        plt.plot([xs[id1-1], xs[id2-1]], [ys[id1-1], ys[id2-1]],
                 color='green')

    plt.grid(True)
    plt.show()

test("/home/one/Downloads/tutor/datasets/sample_dataset1.json")
