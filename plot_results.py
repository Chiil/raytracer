import numpy as np
import matplotlib.pyplot as plt

itot = 256
ktot = 128
dx = 25

x = np.arange(0, itot*dx, dx)
z = np.arange(0, ktot*dx, dx)

surface_down = np.fromfile('surface_down.bin', dtype=np.uint32)
surface_up = np.fromfile('surface_up.bin', dtype=np.uint32)
toa_up = np.fromfile('toa_up.bin', dtype=np.uint32)
toa_down = np.fromfile('toa_down.bin', dtype=np.uint32)
atmos = np.fromfile('atmos.bin', dtype=np.uint32).reshape((ktot, itot))

balance_in = np.int32(toa_down.sum() + surface_up.sum())
balance_out = np.int32(surface_down.sum() + toa_up.sum() + atmos.sum())
balance_net = balance_in - balance_out

print('in: ', balance_in)
print('out: ', balance_out)
print('balance: ', balance_net)

plt.figure()
plt.pcolormesh(x, z, atmos, shading='nearest', cmap=plt.cm.viridis)
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.colorbar()

plt.figure()
plt.plot(x, surface_down / toa_down.mean(), label='surface_down')
plt.plot(x, surface_up / toa_down.mean(), label='surface_up')
plt.plot(x, toa_down / toa_down.mean(), label='toa_down')
plt.plot(x, toa_up / toa_down.mean(), label='toa_up')
plt.legend(loc=0, frameon=False)
plt.xlabel('x (m)')
plt.ylabel('normalized irradiance (-)')

plt.show()
