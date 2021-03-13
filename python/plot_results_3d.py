import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt


# Set the grid and load the data.
itot = 128
jtot = 128
ktot = 64
dx = 50
sigma_x = 40

plot_raw_data = True

x = np.arange(0.5*dx, itot*dx, dx)
y = np.arange(0.5*dx, jtot*dx, dx)
z = np.arange(0.5*dx, ktot*dx, dx)

surface_down_direct = np.fromfile('surface_down_direct.bin', dtype=np.uint64).reshape(jtot, itot)
surface_down_diffuse = np.fromfile('surface_down_diffuse.bin', dtype=np.uint64).reshape(jtot, itot)
surface_down = surface_down_direct + surface_down_diffuse
surface_up = np.fromfile('surface_up.bin', dtype=np.uint64).reshape(jtot, itot)
toa_up = np.fromfile('toa_up.bin', dtype=np.uint64).reshape(jtot, itot)
toa_down = np.fromfile('toa_down.bin', dtype=np.uint64).reshape(jtot, itot)
atmos = np.fromfile('atmos.bin', dtype=np.uint64).reshape((ktot, jtot, itot))


# Check the photon balance.
balance_in = np.int64(toa_down.sum())
balance_out = np.int64(
        surface_down.sum() - surface_up.sum() + toa_up.sum() + atmos.sum())
balance_net = balance_in - balance_out

print('in: ', balance_in)
print('out: ', balance_out)
print('balance: ', balance_net)


# Normalize the data.
norm = toa_down.mean()
surface_down_direct = surface_down_direct / norm
surface_down_diffuse = surface_down_diffuse / norm
surface_down = surface_down / norm
surface_up = surface_up / norm
toa_down = toa_down / norm
toa_up = toa_up / norm
atmos = atmos / norm


# Plot the data.
plt.figure()
plt.pcolormesh(x, z, atmos[:, jtot//2, :], shading='nearest',
        cmap=plt.cm.viridis, vmin=atmos.min(), vmax=atmos.max())
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.title('atmos')

plt.figure()
plt.pcolormesh(x, y, surface_down, shading='nearest',
        cmap=plt.cm.viridis, vmin=surface_down.min(), vmax=surface_down.max())
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('surface_down')

plt.figure()
plt.pcolormesh(x, y, toa_down, shading='nearest',
        cmap=plt.cm.viridis, vmin=surface_down.min(), vmax=surface_down.max())
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('toa_down')

plt.figure()
plt.plot(x, surface_down[jtot//2, :], 'C0-', label='surf_dn')
plt.plot(x, surface_down_direct[jtot//2, :], 'C0--', label='surf_dn_dir')
plt.plot(x, surface_down_diffuse[jtot//2, :], 'C0:', label='surf_dn_dif')
plt.plot(x, surface_up[jtot//2, :], 'C1-', label='surf_up')
plt.plot(x, toa_down[jtot//2, :], 'C2-', label='toa_dn')
plt.plot(x, toa_up[jtot//2, :], 'C3-', label='toa_up')
plt.legend(loc=0, ncol=3)
plt.xlabel('x (m)')
plt.ylabel('normalized irradiance (-)')
plt.ylim(0, 1.3)


# Filter the data
sigma = sigma_x / dx
smooth_data_2d = lambda data : scipy.ndimage.gaussian_filter(data, sigma=sigma, mode=['wrap', 'wrap'])
smooth_data_3d = lambda data : scipy.ndimage.gaussian_filter(data, sigma=sigma, mode=['reflect', 'wrap', 'wrap'])

surface_down_filtered = smooth_data_2d(surface_down)
surface_down_direct_filtered = smooth_data_2d(surface_down_direct)
surface_down_diffuse_filtered = smooth_data_2d(surface_down_diffuse)
surface_up_filtered = smooth_data_2d(surface_up)
toa_down_filtered = smooth_data_2d(toa_down)
toa_up_filtered = smooth_data_2d(toa_up)
atmos_filtered = smooth_data_3d(atmos)


"""
# Plot the filtered data.
plt.figure()
plt.pcolormesh(x, z, atmos_filtered, shading='nearest',
        cmap=plt.cm.viridis, vmin=atmos.min(), vmax=atmos.max())
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('z (m)')
"""

plt.figure()
if plot_raw_data:
    plt.plot(x, surface_down[jtot//2, :], 'k:', linewidth=0.8, alpha=0.5)
    plt.plot(x, surface_down_direct[jtot//2, :], 'k:', linewidth=0.8, alpha=0.5)
    plt.plot(x, surface_down_diffuse[jtot//2, :], 'k:', linewidth=0.8, alpha=0.5)
    plt.plot(x, surface_up[jtot//2, :], 'k:', linewidth=0.8, alpha=0.5)
    plt.plot(x, toa_down[jtot//2, :], 'k:', linewidth=0.8, alpha=0.5)
    plt.plot(x, toa_up[jtot//2, :], 'k:', linewidth=0.8, alpha=0.5)

plt.plot(x, surface_down_filtered[jtot//2, :], 'C0-', label='surf_dn')
plt.plot(x, surface_down_direct_filtered[jtot//2, :], 'C0--', label='surf_dn_dir')
plt.plot(x, surface_down_diffuse_filtered[jtot//2, :], 'C0:', label='surf_dn_dif')
plt.plot(x, surface_up_filtered[jtot//2, :], 'C1-', label='surf_up')
plt.plot(x, toa_down_filtered[jtot//2, :], 'C2-', label='toa_dn')
plt.plot(x, toa_up_filtered[jtot//2, :], 'C3-', label='toa_up')

plt.legend(loc=0, ncol=3)
plt.xlabel('x (m)')
plt.ylabel('normalized irradiance (-)')
plt.ylim(0, 1.3)

plt.show()
