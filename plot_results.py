import numpy as np
import matplotlib.pyplot as plt

itot = 128
ktot = 128

toa_up = np.fromfile('toa_up.bin', dtype=np.uint32)
toa_down = np.fromfile('toa_down.bin', dtype=np.uint32)
surface_down = np.fromfile('surface_down.bin', dtype=np.uint32)
atmos = np.fromfile('atmos.bin', dtype=np.uint32).reshape((ktot, itot))

print('in: ', toa_down.sum())
print('out: ', surface_down.sum() + toa_up.sum() + atmos.sum())
print('balance: ', toa_down.sum() - (surface_down.sum() + toa_up.sum() + atmos.sum()))

plt.figure()
plt.pcolormesh(atmos)
plt.colorbar()

plt.figure()
plt.plot(toa_down, label='toa_down')
plt.plot(toa_up, label='toa_up')
plt.plot(surface_down, label='surface_down')
plt.legend(loc=0, frameon=False)

plt.show()