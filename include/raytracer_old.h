#ifndef RAYTRACER_OLD_H
#define RAYTRACER_OLD_H

void trace_ray(const int ktot, const int itot, const double zsize, const double xsize, 
               const double* k_ext, const double* ssa, const double* g,
               const double dx_grid, const double albedo, const double sza_rad,
               const double k_ext_gas, const double k_null,
               const int n_photon, unsigned int* sfc_dir, unsigned int* sfc_dif, unsigned int* atmos_count);

#endif // RAYTRACER_OLD_H
