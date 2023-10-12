import os
import shutil

import numpy as np

from Deformation3D import prepare_vega_files, deform_models, get_models_pcd
# Deformation3D.hello()


config_filepath = "./corn_vox.configs"
txt_filepath = "demo-7.txt"
vega_dir = "veg_dir/"
out_dir = "out_dir/"

leaf_num = 7  # Number of instances
new_num = 10  # Number of deformations
integrator_times = 10  # leaf integration time
stem_integrator_times = 7  # stem integration time

# Minimum value of force exerted by plant organs (sorted by organ instance)
min_base_force = np.array([
    [-5, -5, -20],  # stem
    [-5, -5, -15],  # leaf 1
    [-5, -5, -15],  # leaf 2
    [-5, -5, -15],  # leaf 3
    [-5, -5, -15],  # leaf 4
    [-5, -5, -15],  # leaf 5
    [-5, -5, -15],  # leaf 6
    [-5, -5, -5],   # leaf 7
])

# Maixmum value of force exerted by plant organs (sorted by organ instance)
max_base_force = np.array([
    [5, 5, 0],  # stem
    [5, 5, 15],  # leaf 1
    [5, 5, 15],  # leaf 2
    [5, 5, 15],  # leaf 3
    [5, 5, 15],  # leaf 4
    [5, 5, 15],  # leaf 5
    [5, 5, 15],  # leaf 6
    [5, 5, 5],   # leaf 7
])

TMP_DIR = 'tmp'
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(vega_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

# Create the files required for deformation (only need to be executed once)
prepare_vega_files(txt_filepath, vega_dir, leaf_num, voxel_size=0.02, MATERIALS="\n*MATERIAL STEM\nENU, 100, 100000, 0.01\n\n*MATERIAL LEAF\nENU, 100, 1000000, 0.45\n")


print("prepare_vega_files done ")
# Start deformation
deform_models(config_filepath, txt_filepath, vega_dir, out_dir, leaf_num, new_num,
              min_base_force, max_base_force,
              integrator_times=integrator_times, stem_integrator_times=stem_integrator_times)

print("deform_models done ")
# Interpolate point cloud from obj vertices
get_models_pcd(out_dir, vega_dir, remove_outlier=True, transform_axis=True, normal=False)


if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
    print(f"Folder '{TMP_DIR}' and its contents have been successfully deleted.")
else:
    print(f"Folder '{TMP_DIR}' does not exist.")