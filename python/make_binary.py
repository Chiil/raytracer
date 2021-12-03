import netCDF4 as nc
import numpy as np

ngpt = 2

ncfile = nc.Dataset("cabauw_output_res.nc", "r")
k_ext_cloud = ncfile.variables["kext_cloud"][1, :, :, :]
k_ext_gas = ncfile.variables["kext_gas"][1, :, :, :]
ssa = ncfile.variables["ssa"][1, :, :, :]
asy = ncfile.variables["g"][1, :, :, :]

k_ext_cloud.data.tofile("k_ext_cloud.bin")
k_ext_gas.data.tofile("k_ext_gas.bin")
ssa.data.tofile("ssa.bin")
asy.data.tofile("asy.bin")
