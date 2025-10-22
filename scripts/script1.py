import numpy as np

data=np.load(r'E:\someShy\ML_Practice\scripts\mushroom.npy', allow_pickle=True)

print(data[:10,:6])