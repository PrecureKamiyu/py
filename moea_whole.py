import moea_pymoo
import json as js


ns = [10,20,30]
for n in ns:
    data = moea_pymoo.main(
        number_of_servers=n,
        number_of_moea_gen=40,
        path_to_saved_result=f"./records/moea_{n}_servers_front.npy",
    )
    path = f"./records/moea_{n}_servers.json"
    with open(path, 'w') as js_file:
        js.dump(data, js_file)
