import nsga_pymoo
import json as js

if __name__ == "__main__":
    ns = [10, 20, 30]
    for n in ns:
        data = nsga_pymoo.main(
            number_of_servers=n,
            number_of_generation=40,
        )
        path = f"./records/nsga_{n}_servers.json"
        with open(path, 'w') as js_file:
            js.dump(data, js_file)
