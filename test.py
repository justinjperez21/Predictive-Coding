from model import *
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

widths = [5,7,10]
clamped_layers = [2]
pcmodel = PC_Model(widths, update_rate=0.001, clamped_layers=clamped_layers)
pcmodel.set_activity(2, np.array([0 for i in range(10)]))

energy_data = []
for i in tqdm(range(10000)):
    curr_energy = pcmodel.get_error()**2
    energy_data.append(np.log(curr_energy))
    pcmodel.update_activity()
    pcmodel.update_prediction()

energy_data.append(np.log(pcmodel.get_error()**2))
plt.plot(energy_data)
plt.show()
