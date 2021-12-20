import netCDF4 as nc
import numpy as np
import subprocess
import re
import matplotlib.pyplot as plt
from progress.bar import Bar

ngpt = 34
TF = np.float32

durations = np.zeros(ngpt)
bar = Bar('Processing', max=ngpt)

for i in range(ngpt):
    ncfile = nc.Dataset("cabauw_output_res.nc", "r")

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
    bar.next()

print()
print('Total duration ', durations.sum(), ' (s)')

plt.figure()
plt.bar(np.arange(ngpt), durations)
plt.xlabel('gpt')
plt.ylabel('duration (s)')
plt.show()
