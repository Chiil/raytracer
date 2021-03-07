import numpy as np
import matplotlib.pyplot as plt

itot = 256
ktot = 128
dx = 25

x = np.arange(0, itot*dx, dx)
z = np.arange(0, ktot*dx, dx)

toa_up = np.fromfile('toa_up.bin', dtype=np.uint32)
toa_down = np.fromfile('toa_down.bin', dtype=np.uint32)
surface_down = np.fromfile('surface_down.bin', dtype=np.uint32)
atmos = np.fromfile('atmos.bin', dtype=np.uint32).reshape((ktot, itot))

print('in: ', toa_down.sum())
print('out: ', surface_down.sum() + toa_up.sum() + atmos.sum())
print('balance: ', toa_down.sum() - (surface_down.sum() + toa_up.sum() + atmos.sum()))

plt.figure()
plt.pcolormesh(x, z, atmos, shading='nearest', cmap=plt.cm.viridis)
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.colorbar()

plt.figure()
plt.plot(x, toa_down / toa_down.mean(), label='toa_down')
plt.plot(x, toa_up / toa_down.mean(), label='toa_up')
plt.plot(x, surface_down / toa_down.mean(), label='surface_down')
plt.legend(loc=0, frameon=False)
plt.xlabel('x (m)')
plt.ylabel('normalized irradiance (-)')

plt.show()
