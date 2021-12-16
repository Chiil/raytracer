import netCDF4 as nc
import numpy as np
TF = np.float32

ngpt = 19

ncfile = nc.Dataset("cabauw_output_res.nc", "r")
k_ext_cloud = ncfile.variables["kext_cloud"][ngpt, :, :, :].astype(TF)
k_ext_gas = ncfile.variables["kext_gas"][ngpt, :, :, :].astype(TF)
ssa = ncfile.variables["ssa"][ngpt, :, :, :].astype(TF)
asy = ncfile.variables["g"][ngpt, :, :, :].astype(TF)

## Remove 'cirrus' layer that was put in the upper for previous testing
asy[265:269,:,:] = 0
ssa[265:269,:,:] = ssa[264,:,:]
k_ext_cloud[265:269,:,:] = 0

k_ext_cloud.data.tofile("k_ext_cloud.bin")
k_ext_gas.data.tofile("k_ext_gas.bin")
ssa.data.tofile("ssa.bin")
asy.data.tofile("asy.bin")
