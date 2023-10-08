import os

data_root = "test_results/single"

new_res = "name,total_all_ap,total_all_ap_50%,total_all_ap_25%,stem_all_ap,stem_all_ap_50%,stem_all_ap_25%,leaf_all_ap,leaf_all_ap_50%,leaf_all_ap_25%\n"
for item in os.listdir(data_root):
    with open(os.path.join(data_root, item), 'r') as f:
        line = f.readlines()[1]
        new_res += f"{item[:-13]}{line[2:]}"

with open(os.path.join(data_root, "total.csv"), 'w') as f:
    f.write(new_res)