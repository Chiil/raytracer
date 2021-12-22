import netCDF4 as nc
import numpy as np
import subprocess
import re
import matplotlib.pyplot as plt
from progress.bar import Bar
dz = 30.
cp = 1004.
rho= 1 #assume constant density of 1, this is obviously wrong in the top of the domain but order of magnitude of HR is fine
ngpt = 34
TF = np.float32

durations = np.zeros(ngpt)
bar = Bar('Processing', max=ngpt)

## input TOD irradiance is based on different zenith angle than we use here, correct for that
sza_cur = 50
sza_ref = 37.908
tod_frac = np.cos(np.deg2rad(sza_cur)) / np.cos(np.deg2rad(sza_ref))

# output arrays
direct = np.zeros((240,240))
diffuse = np.zeros((240,240))
heating = np.zeros((284,240,240))

tod_min = 0 # only run if TOD irradiance is higher to save some time
for i in range(ngpt):
    ncfile = nc.Dataset("cabauw_output_res.nc", "r")
    tod_dir = ncfile.variables["tod_dir"][i].astype(TF)
    tod_dif = ncfile.variables["tod_dif"][i].astype(TF)
    tod_tot = tod_dir+tod_dif
    if tod_tot > tod_min:
    
        k_ext_cloud = ncfile.variables["kext_cloud"][i, :, :, :].astype(TF)
        k_ext_gas = ncfile.variables["kext_gas"][i, :, :, :].astype(TF)
        ssa = ncfile.variables["ssa"][i, :, :, :].astype(TF)
        asy = ncfile.variables["g"][i, :, :, :].astype(TF)
    
        asy[265:269,:,:] = 0
        ssa[265:269,:,:] = ssa[264,:,:]
        k_ext_cloud[265:269,:,:] = 0
        
        k_ext_cloud.data.tofile("k_ext_cloud.bin")
        k_ext_gas.data.tofile("k_ext_gas.bin")
        ssa.data.tofile("ssa.bin")
        asy.data.tofile("asy.bin")
    
        proc = subprocess.run(['./raytracer_3d_io_gpu', '25'], stdout = subprocess.PIPE)
        s = proc.stdout.decode('utf-8')
        duration = float(re.findall(r'\b\d+\.*\d*\b', s)[-1])
        durations[i] = duration
    
        toa_down = np.fromfile("toa_down.bin",dtype=TF).mean()

        abs_dir = np.fromfile("atmos_direct.bin",dtype=TF).reshape(284,240,240) 
        abs_dif = np.fromfile("atmos_diffuse.bin",dtype=TF).reshape(284,240,240)
        abs_tot = (abs_dir + abs_dif) / toa_down * tod_tot * tod_frac
        heating += abs_tot / (rho * dz * cp) * 86400
    
        sfc_dir = np.fromfile("surface_down_direct.bin",dtype=TF).reshape(240,240)
        sfc_dif = np.fromfile("surface_down_diffuse.bin",dtype=TF).reshape(240,240)
        
        direct += sfc_dir / toa_down * tod_tot * tod_frac 
        diffuse += sfc_dif / toa_down * tod_tot * tod_frac 
        bar.next()

print()
print('Total duration ', durations.sum(), ' (s)')

## save heating and surface arrays to netCDF
ncf = nc.Dataset("heating_surface.nc", "w")
ncf.createDimension("z", 284)
ncf.createDimension("y", 240)
ncf.createDimension("x", 240)
z = ncf.createVariable("z", "f4", ("z",))
y = ncf.createVariable("y", "f4", ("y",))
x = ncf.createVariable("x", "f4", ("x",))
z[:] = np.arange(284) * 30 + 15
y[:] = np.arange(240) * 100 + 50
x[:] = np.arange(240) * 100 + 50

nc_dir = ncf.createVariable("direct", "f4", ("y", "x")) 
nc_dif = ncf.createVariable("diffuse", "f4", ("y", "x")) 
nc_tot = ncf.createVariable("global", "f4", ("y", "x")) 
nc_hr  = ncf.createVariable("heating", "f4", ("z","y", "x")) 

nc_dir[:] = direct
nc_dif[:] = diffuse
nc_tot[:] = direct+diffuse
nc_hr[:] = heating

plt.figure()
plt.bar(np.arange(ngpt), durations)
plt.xlabel('gpt')
plt.ylabel('duration (s)')
plt.draw()

xy = np.linspace(0,24.1,240)
z = np.linspace(0,8.52,284)

plt.figure()
plt.pcolormesh(xy,xy,direct+diffuse)
plt.colorbar(label="W/m2")
plt.xlabel("x [km]")
plt.ylabel("y [km]")
plt.title("Surface irradiance")
plt.draw()

plt.figure()
plt.pcolormesh(xy,z,heating[:,40,:])
plt.colorbar(label="K/d")
plt.xlabel("x [km]")
plt.ylabel("y [km]")
plt.title("Heating rates")
plt.show()

