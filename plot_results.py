import numpy as np
import matplotlib.pyplot as plt

itot = 128
ktot = 128

surface = np.fromfile('surface.bin', dtype=np.int32)
toa = np.fromfile('toa.bin', dtype=np.int32)
atmos = np.fromfile('atmos.bin', dtype=np.int32).reshape((ktot, itot))

plt.figure()
plt.subplot(211)
plt.pcolormesh(atmos)
plt.colorbar()
plt.subplot(212)
plt.plot(surface)
plt.plot(toa)
plt.show()
