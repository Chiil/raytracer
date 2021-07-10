#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>

#include <curand_kernel.h>

#define uint64_t unsigned long long

struct Vector
{
    double x;
    double y;
    double z;
};


Vector cross(const Vector& v1, const Vector& v2)
{
    return Vector{
            v1.y*v2.z - v1.z*v2.y,
            v1.z*v2.x - v1.x*v2.z,
            v1.x*v2.y - v1.y*v2.x};
}


double dot(const Vector& v1, const Vector& v2)
{
    return v1.x*v2.x + v1.y*v2.y + v1.z*v1.z;
}


double norm(const Vector& v) { return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }


Vector normalize(const Vector& v)
{
    const double length = norm(v);
    return Vector{ v.x/length, v.y/length, v.z/length};
}


Vector operator*(const Vector& v, const double s) { return Vector{s*v.x, s*v.y, s*v.z}; }
Vector operator*(const double s, const Vector& v) { return Vector{s*v.x, s*v.y, s*v.z}; }
Vector operator-(const Vector& v1, const Vector& v2) { return Vector{v1.x-v2.x, v1.y-v2.y, v1.z-v2.z}; }
Vector operator+(const Vector& v1, const Vector& v2) { return Vector{v1.x+v2.x, v1.y+v2.y, v1.z+v2.z}; }


enum class Photon_kind { Direct, Diffuse };
enum class Photon_status { Enabled, Disabled };


struct Photon
{
    Vector position;
    Vector direction;
    Photon_kind kind;
    Photon_status status;
};


double pow2(const double d) { return d*d; }


#define cuda_safe_call(ans) { gpu_assert((ans), __FILE__, __LINE__); }

inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


template<typename T>
T* allocate_gpu(const int length)
{
    T* data_ptr = nullptr;
    cuda_safe_call(cudaMalloc((void **) &data_ptr, length*sizeof(T)));

    return data_ptr;
}


template<typename T>
void free_gpu(T*& data_ptr)
{
    cuda_safe_call(cudaFree(data_ptr));
    data_ptr = nullptr;
}


template<typename T>
void copy_to_gpu(T* gpu_data, const T* cpu_data, const int length)
{
    cuda_safe_call(cudaMemcpy(gpu_data, cpu_data, length*sizeof(T), cudaMemcpyHostToDevice));
}


template<typename T>
void copy_from_gpu(T* cpu_data, const T* gpu_data, const int length)
{
    cuda_safe_call(cudaMemcpy(cpu_data, gpu_data, length*sizeof(T), cudaMemcpyDeviceToHost));
}


double rayleigh(const double random_number)
{
    const double q = 4.*random_number - 2.;
    const double d = 1. + pow2(q);
    const double u = std::pow(-q + sqrt(d), 1./3.);
    return u - 1./u;
}


double henyey(const double g, const double random_number)
{
    const double a = pow2(1. - pow2(g));
    const double b = 2.*g*pow2(2.*random_number*g + 1. - g);
    const double c = -g/2. - 1./(2.*g);
    return -1.*(a/b) - c;
}


double sample_tau(const double random_number)
{
    return -1.*std::log(-random_number + 1.) + std::numeric_limits<double>::epsilon();
}


__host__ __device__
void reset_photon(
        Photon& photon,
        const double random_number_x, const double random_number_y,
        const double x_size, const double y_size, const double z_size,
        const double zenith_angle, const double azimuth_angle)
{
    photon.position.x = x_size * random_number_x;
    photon.position.y = y_size * random_number_y;
    photon.position.z = z_size;
    photon.direction.x = -std::sin(zenith_angle)*std::cos(azimuth_angle);
    photon.direction.y = -std::sin(zenith_angle)*std::sin(azimuth_angle);
    photon.direction.z = -std::cos(zenith_angle);
    photon.kind = Photon_kind::Direct;
    photon.status = Photon_status::Enabled;
}


template<typename T>
struct Random_number_generator
{
    __device__
    Random_number_generator(unsigned int tid)
    {
        curand_init(tid, tid, 0, &state);
    }

    __device__ T
    operator ()(void)
    {
        return curand_uniform(&state);
    }

    curandState state;
};


__global__
void ray_tracer_init_kernel(
        Photon* __restrict__ photons,
        uint64_t* __restrict__ toa_down_count,
        double x_size, double y_size, double z_size,
        double dx_grid, double dy_grid, double dz_grid,
        double zenith_angle, double azimuth_angle, 
        const int itot)
{
    const int n = blockIdx.x*blockDim.x + threadIdx.x;

    Random_number_generator<double> rng(n);

    reset_photon(
            photons[n], rng(), rng(),
            x_size, y_size, z_size,
            zenith_angle, azimuth_angle);

    const int i = photons[n].position.x / dx_grid;
    const int j = photons[n].position.y / dy_grid;
    const int ij = i + j*itot;

    // Make sure increment is atomic.
    atomicAdd(&toa_down_count[ij], 1);
}


__global__
void ray_tracer_kernel(
        Photon* __restrict__ photons,
        uint64_t* __restrict__ toa_down_count,
        double x_size, double y_size, double z_size,
        double dx_grid, double dy_grid, double dz_grid,
        double zenith_angle, double azimuth_angle, 
        const int itot)
{
    const int n = blockIdx.x*blockDim.x + threadIdx.x;

    Random_number_generator<double> rng(n);

    while (true)
    {
        const double dn = sample_tau(rng()) / k_ext_null;
        double dx = photons[n].direction.x * dn;
        double dy = photons[n].direction.y * dn;
        double dz = photons[n].direction.z * dn;

        bool surface_exit = false;
        bool toa_exit = false;

        if ((photons[n].position.z + dz) <= 0.)
        {
            const double fac = std::abs(photons[n].position.z / dz);
            dx *= fac;
            dy *= fac;
            dz *= fac;

            surface_exit = true;
        }
        else if ((photons[n].position.z + dz) >= z_size)
        {
            const double fac = std::abs((z_size - photons[n].position.z) / dz);
            dx *= fac;
            dy *= fac;
            dz *= fac;

            toa_exit = true;
        }

        photons[n].position.x += dx;
        photons[n].position.y += dy;
        photons[n].position.z += dz;

        // Cyclic boundary condition in x.
        photons[n].position.x = std::fmod(photons[n].position.x, x_size);
        if (photons[n].position.x < 0.)
            photons[n].position.x += x_size;

        // Cyclic boundary condition in y.
        photons[n].position.y = std::fmod(photons[n].position.y, y_size);
        if (photons[n].position.y < 0.)
            photons[n].position.y += y_size;

        // Handle the surface and top exits.
        const int i = photons[n].position.x / dx_grid;
        const int j = photons[n].position.y / dy_grid;
        const int ij = i + j*itot;

        if (surface_exit)
        {
            if (photons[n].kind == Photon_kind::Direct
                    && photons[n].status == Photon_status::Enabled)
            {
                ++n_photons_out;
                atomicAdd(&surface_down_direct_count[ij], 1)
            }
            else if (photons[n].kind == Photon_kind::Diffuse
                    && photons[n].status == Photon_status::Enabled)
            {
                ++n_photons_out;
                atomicAdd(&surface_down_diffuse_count[ij], 1)
            }

            // Surface scatter if smaller than albedo, otherwise absorb
            if (rg.fp64() <= surface_albedo)
            {
                if (photons[n].status == Photon_status::Enabled)
                {
                    --n_photons_out;
                    atomicAdd(&surface_up_count[ij], 1)
                }

                const double mu_surface = std::sqrt(rng());
                const double azimuth_surface = 2.*M_PI*rng();
                // CvH: is this correct?
                photons[n].direction.x = mu_surface*std::sin(azimuth_surface);
                photons[n].direction.y = mu_surface*std::cos(azimuth_surface);
                photons[n].direction.z = std::sqrt(1. - mu_surface*mu_surface);
                photons[n].kind = Photon_kind::Diffuse;
            }
            else
            {
                reset_photon(
                        photons[n], rng(), rng(),
                        x_size, y_size, z_size,
                        zenith_angle, azimuth_angle);

                if (photon_generation_completed)
                    photons[n].status = Photon_status::Disabled;

                if (photons[n].status == Photon_status::Enabled)
                {
                    ++n_photons_in;

                    const int i_new = photons[n].position.x / dx_grid;
                    const int j_new = photons[n].position.y / dy_grid;
                    const int ij_new = i_new + j_new*itot;

                    atomicAdd(&toa_down_count[ij_new], 1)
                }
            }
        }
        else if (toa_exit)
        {
            if (photons[n].status == Photon_status::Enabled)
            {
                ++n_photons_out;
                atomicAdd(&toa_up_count[ij_new], 1)
            }

            reset_photon(
                    photons[n], rng(), rng(),
                    x_size, y_size, z_size,
                    zenith_angle, azimuth_angle);


            if (photon_generation_completed)
                photons[n].status = Photon_status::Disabled;

            if (photons[n].status == Photon_status::Enabled)
            {
                ++n_photons_in;

                const int i_new = photons[n].position.x / dx_grid;
                const int j_new = photons[n].position.y / dy_grid;
                const int ij_new = i_new + j_new*itot;

                atomicAdd(&toa_down_count[ij_new], 1)
            }
        }
        else
        {
            break;
        }
    }
}


void run_ray_tracer(const uint64_t n_photons)
{
    //// DEFINE INPUT ////
    // Grid properties.
    const double dx_grid = 50.;
    const double dy_grid = 50.;
    const double dz_grid = 25.;

    const int itot = 128;
    const int jtot = 128;
    const int ktot = 128;

    const double x_size = itot*dx_grid;
    const double y_size = jtot*dy_grid;
    const double z_size = ktot*dz_grid;

    // Radiation properties.
    const double surface_albedo = 0.2;
    const double zenith_angle = 50.*(M_PI/180.);
    const double azimuth_angle = 20.*(M_PI/180.);

    // Input fields.
    const double k_ext_gas = 1.e-4; // 3.e-4;
    const double ssa_gas = 0.5;
    const double asy_gas = 0.;

    const double k_ext_cloud = 5.e-3;
    const double ssa_cloud = 0.9;
    const double asy_cloud = 0.85;

    // Create the spatial fields.
    std::vector<double> k_ext(itot*jtot*ktot);
    std::vector<double> ssa(itot*jtot*ktot);
    std::vector<double> asy(itot*jtot*ktot);

    // First add the gases over the entire domain.
    std::fill(k_ext.begin(), k_ext.end(), k_ext_gas);
    std::fill(ssa.begin(), ssa.end(), ssa_gas);

    // Add a block cloud.
    for (int k=0; k<ktot; ++k)
        for (int j=0; j<jtot; ++j)
            for (int i=0; i<itot; ++i)
            {
                if (  (i+0.5)*dx_grid > 4000. && (i+0.5)*dx_grid < 5000.
                   && (j+0.5)*dy_grid > 2700. && (j+0.5)*dy_grid < 3700.
                   && (k+0.5)*dz_grid > 1000. && (k+0.5)*dz_grid < 1500.)
                {
                    const int ijk = i + j*itot + k*itot*jtot;
                    k_ext[ijk] = k_ext_gas + k_ext_cloud;
                    ssa[ijk] = (ssa_gas*k_ext_gas + ssa_cloud*k_ext_cloud)
                             / (k_ext_gas + k_ext_cloud);
                    asy[ijk] = (asy_gas*ssa_gas*k_ext_gas + asy_cloud*ssa_cloud*k_ext_cloud)
                             / (ssa_gas*k_ext_gas + ssa_cloud*k_ext_cloud);
                }
            }

    // Set the step size for the transport solver to the maximum extinction coefficient.
    const double k_ext_null = *std::max_element(k_ext.begin(), k_ext.end());


    //// PREPARE OUTPUT ARRAYS ////
    std::vector<uint64_t> surface_down_direct_count(itot*jtot);
    std::vector<uint64_t> surface_down_diffuse_count(itot*jtot);
    std::vector<uint64_t> surface_up_count(itot*jtot);
    std::vector<uint64_t> toa_down_count(itot*jtot);
    std::vector<uint64_t> toa_up_count(itot*jtot);
    std::vector<uint64_t> atmos_direct_count(itot*jtot*ktot);
    std::vector<uint64_t> atmos_diffuse_count(itot*jtot*ktot);


    //// COPY THE DATA TO THE GPU.
    // Input array.
    double* k_ext_gpu = allocate_gpu<double>(itot*jtot*ktot);
    double* ssa_gpu = allocate_gpu<double>(itot*jtot*ktot);
    double* asy_gpu = allocate_gpu<double>(itot*jtot*ktot);

    copy_to_gpu(k_ext_gpu, k_ext.data(), itot*jtot*ktot);
    copy_to_gpu(ssa_gpu, ssa.data(), itot*jtot*ktot);
    copy_to_gpu(asy_gpu, asy.data(), itot*jtot*ktot);

    // Output arrays. Copy them in order to enable restarts later.
    uint64_t* surface_down_direct_count_gpu = allocate_gpu<uint64_t>(itot*jtot);
    uint64_t* surface_down_diffuse_count_gpu = allocate_gpu<uint64_t>(itot*jtot);
    uint64_t* surface_up_count_gpu = allocate_gpu<uint64_t>(itot*jtot);
    uint64_t* toa_down_count_gpu = allocate_gpu<uint64_t>(itot*jtot);
    uint64_t* toa_up_count_gpu = allocate_gpu<uint64_t>(itot*jtot);
    uint64_t* atmos_direct_count_gpu = allocate_gpu<uint64_t>(itot*jtot*ktot);
    uint64_t* atmos_diffuse_count_gpu = allocate_gpu<uint64_t>(itot*jtot*ktot);

    copy_to_gpu(surface_down_direct_count_gpu, surface_down_direct_count.data(), itot*jtot);
    copy_to_gpu(surface_down_diffuse_count_gpu, surface_down_diffuse_count.data(), itot*jtot);
    copy_to_gpu(surface_up_count_gpu, surface_up_count.data(), itot*jtot);
    copy_to_gpu(toa_down_count_gpu, toa_down_count.data(), itot*jtot);
    copy_to_gpu(toa_up_count_gpu, toa_up_count.data(), itot*jtot);
    copy_to_gpu(atmos_direct_count_gpu, atmos_direct_count.data(), itot*jtot*ktot);
    copy_to_gpu(atmos_diffuse_count_gpu, atmos_diffuse_count.data(), itot*jtot*ktot);


    //// RUN THE RAY TRACER ////
    constexpr int n_photons_batch = 256;
    Photon* photons = allocate_gpu<Photon>(n_photons_batch);

    dim3 grid{1}, block{n_photons_batch};

    auto start = std::chrono::high_resolution_clock::now();

    ray_tracer_init_kernel<<<grid, block>>>(
            photons,
            toa_down_count_gpu,
            x_size, y_size, z_size,
            dx_grid, dy_grid, dz_grid,
            zenith_angle, azimuth_angle,
            itot);

    /*
    uint64_t n_photons_in = n_photons_batch;
    uint64_t n_photons_out = 0;

    #pragma omp parallel
    {
        #ifdef _OPENMP
        Random_number_generator rg(omp_get_thread_num());
        #else
        Random_number_generator rg(static_cast<uint64_t>(3849284923));
        #endif

        #pragma omp for
        for (int n=0; n<n_photons_batch; ++n)
        {
            reset_photon(
                    photons[n], rg.fp64(), rg.fp64(),
                    x_size, y_size, z_size,
                    zenith_angle, azimuth_angle);

            const int i = photons[n].position.x / dx_grid;
            const int j = photons[n].position.y / dy_grid;
            const int ij = i + j*itot;

            #pragma omp atomic
            ++toa_down_count[ij];
        }

        while ((n_photons_in < n_photons) || (n_photons_in > n_photons_out))
        {
            const bool photon_generation_completed = n_photons_in >= n_photons;

            // Transport the photons
            #pragma omp for reduction(+:n_photons_in) reduction(+:n_photons_out)
            for (int n=0; n<n_photons_batch; ++n)
            {
                while (true)
                {
                    const double dn = sample_tau(rg.fp64()) / k_ext_null;
                    double dx = photons[n].direction.x * dn;
                    double dy = photons[n].direction.y * dn;
                    double dz = photons[n].direction.z * dn;

                    bool surface_exit = false;
                    bool toa_exit = false;

                    if ((photons[n].position.z + dz) <= 0.)
                    {
                        const double fac = std::abs(photons[n].position.z / dz);
                        dx *= fac;
                        dy *= fac;
                        dz *= fac;

                        surface_exit = true;
                    }
                    else if ((photons[n].position.z + dz) >= z_size)
                    {
                        const double fac = std::abs((z_size - photons[n].position.z) / dz);
                        dx *= fac;
                        dy *= fac;
                        dz *= fac;

                        toa_exit = true;
                    }

                    photons[n].position.x += dx;
                    photons[n].position.y += dy;
                    photons[n].position.z += dz;

                    // Cyclic boundary condition in x.
                    photons[n].position.x = std::fmod(photons[n].position.x, x_size);
                    if (photons[n].position.x < 0.)
                        photons[n].position.x += x_size;

                    // Cyclic boundary condition in y.
                    photons[n].position.y = std::fmod(photons[n].position.y, y_size);
                    if (photons[n].position.y < 0.)
                        photons[n].position.y += y_size;

                    // Handle the surface and top exits.
                    const int i = photons[n].position.x / dx_grid;
                    const int j = photons[n].position.y / dy_grid;
                    const int ij = i + j*itot;

                    if (surface_exit)
                    {
                        if (photons[n].kind == Photon_kind::Direct
                                && photons[n].status == Photon_status::Enabled)
                        {
                            ++n_photons_out;

                            #pragma omp atomic
                            ++surface_down_direct_count[ij];
                        }
                        else if (photons[n].kind == Photon_kind::Diffuse
                                && photons[n].status == Photon_status::Enabled)
                        {
                            ++n_photons_out;

                            #pragma omp atomic
                            ++surface_down_diffuse_count[ij];
                        }

                        // Surface scatter if smaller than albedo, otherwise absorb
                        if (rg.fp64() <= surface_albedo)
                        {
                            if (photons[n].status == Photon_status::Enabled)
                            {
                                --n_photons_out;

                                #pragma omp atomic
                                ++surface_up_count[ij];
                            }

                            const double mu_surface = std::sqrt(rg.fp64());
                            const double azimuth_surface = 2.*M_PI*rg.fp64();
                            // CvH: is this correct?
                            photons[n].direction.x = mu_surface*std::sin(azimuth_surface);
                            photons[n].direction.y = mu_surface*std::cos(azimuth_surface);
                            photons[n].direction.z = std::sqrt(1. - mu_surface*mu_surface);
                            photons[n].kind = Photon_kind::Diffuse;
                        }
                        else
                        {
                            reset_photon(
                                    photons[n], rg.fp64(), rg.fp64(),
                                    x_size, y_size, z_size,
                                    zenith_angle, azimuth_angle);

                            if (photon_generation_completed)
                                photons[n].status = Photon_status::Disabled;

                            if (photons[n].status == Photon_status::Enabled)
                            {
                                ++n_photons_in;

                                const int i_new = photons[n].position.x / dx_grid;
                                const int j_new = photons[n].position.y / dy_grid;
                                const int ij_new = i_new + j_new*itot;
                                #pragma omp atomic
                                ++toa_down_count[ij_new];
                            }
                        }
                    }
                    else if (toa_exit)
                    {
                        if (photons[n].status == Photon_status::Enabled)
                        {
                            ++n_photons_out;

                            #pragma omp atomic
                            ++toa_up_count[ij];
                        }

                        reset_photon(
                                photons[n], rg.fp64(), rg.fp64(),
                                x_size, y_size, z_size,
                                zenith_angle, azimuth_angle);


                        if (photon_generation_completed)
                            photons[n].status = Photon_status::Disabled;

                        if (photons[n].status == Photon_status::Enabled)
                        {
                            ++n_photons_in;

                            const int i_new = photons[n].position.x / dx_grid;
                            const int j_new = photons[n].position.y / dy_grid;
                            const int ij_new = i_new + j_new*itot;
                            #pragma omp atomic
                            ++toa_down_count[ij_new];
                        }
                    }
                    else
                    {
                        break;
                    }
                }
            }

            // Handle the collision events.
            #pragma omp for reduction(+:n_photons_in) reduction(+:n_photons_out)
            for (int n=0; n<n_photons_batch; ++n)
            {
                const int i = photons[n].position.x / dx_grid;
                const int j = photons[n].position.y / dy_grid;
                const int k = photons[n].position.z / dz_grid;

                const int ijk = i + j*itot + k*itot*jtot;

                const double random_number = rg.fp64();

                // Null collision.
                if (random_number >= (k_ext[ijk] / k_ext_null))
                {
                }
                // Scattering.
                else if (random_number <= ssa[ijk] * k_ext[ijk] / k_ext_null)
                {
                    const bool cloud_scatter = rg.fp64() < (k_ext[ijk] - k_ext_gas) / k_ext[ijk];
                    const double cos_scat = cloud_scatter ? henyey(asy[ijk], rg.fp64()) : rayleigh(rg.fp64());
                    const double sin_scat = std::sqrt(1. - cos_scat*cos_scat);

                    Vector t1{0., 0., 0.};
                    if (std::fabs(photons[n].direction.x) < std::fabs(photons[n].direction.y))
                    {
                        if (std::fabs(photons[n].direction.x) < std::fabs(photons[n].direction.z))
                            t1.x = 1;
                        else
                            t1.z = 1;
                    }
                    else
                    {
                        if (std::fabs(photons[n].direction.y) < std::fabs(photons[n].direction.z))
                            t1.y = 1;
                        else
                            t1.z = 1;
                    }
                    t1 = normalize(t1 - photons[n].direction*dot(t1, photons[n].direction));
                    Vector t2 = cross(photons[n].direction, t1);

                    const double phi = 2.*M_PI*rg.fp64();

                    photons[n].direction = cos_scat*photons[n].direction
                            + sin_scat*(std::sin(phi)*t1 + std::cos(phi)*t2);

                    photons[n].kind = Photon_kind::Diffuse;
                }
                // Absorption.
                else
                {
                    if (photons[n].status == Photon_status::Enabled)
                    {
                        ++n_photons_out;

                        #pragma omp atomic
                        ++atmos_count[ijk];
                    }

                    reset_photon(
                            photons[n], rg.fp64(), rg.fp64(),
                            x_size, y_size, z_size,
                            zenith_angle, azimuth_angle);

                    if (photon_generation_completed)
                        photons[n].status = Photon_status::Disabled;

                    if (photons[n].status == Photon_status::Enabled)
                    {
                        ++n_photons_in;

                        const int i_new = photons[n].position.x / dx_grid;
                        const int j_new = photons[n].position.y / dy_grid;
                        const int ij_new = i_new + j_new*itot;
                        #pragma omp atomic
                        ++toa_down_count[ij_new];
                    }
                }
            }
        }
    }
    */

    cuda_safe_call(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "Duration: " << std::setprecision(5) << duration << " (s)" << std::endl;
    //// END RUNNING OF RAY TRACER ////


    //// COPY OUTPUT BACK TO CPU ////
    copy_from_gpu(surface_down_direct_count.data(), surface_down_direct_count_gpu, itot*jtot);
    copy_from_gpu(surface_down_diffuse_count.data(), surface_down_diffuse_count_gpu, itot*jtot);
    copy_from_gpu(surface_up_count.data(), surface_up_count_gpu, itot*jtot);
    copy_from_gpu(toa_down_count.data(), toa_down_count_gpu, itot*jtot);
    copy_from_gpu(toa_up_count.data(), toa_up_count_gpu, itot*jtot);
    copy_from_gpu(atmos_direct_count.data(), atmos_direct_count_gpu, itot*jtot*ktot);
    copy_from_gpu(atmos_diffuse_count.data(), atmos_diffuse_count_gpu, itot*jtot*ktot);


    //// SAVE THE OUTPUT TO DISK ////
    auto save_binary = [](const std::string& name, void* ptr, const int size)
    {
        std::ofstream binary_file(name + ".bin", std::ios::out | std::ios::trunc | std::ios::binary);

        if (binary_file)
            binary_file.write(reinterpret_cast<const char*>(ptr), size*sizeof(uint64_t));
        else
        {
            std::string error = "Cannot write file \"" + name + ".bin\"";
            throw std::runtime_error(error);
        }
    };

    save_binary("toa_down", toa_down_count.data(), itot*jtot);
    save_binary("toa_up", toa_up_count.data(), itot*jtot);
    save_binary("surface_down_direct", surface_down_direct_count.data(), itot*jtot);
    save_binary("surface_down_diffuse", surface_down_diffuse_count.data(), itot*jtot);
    save_binary("surface_up", surface_up_count.data(), itot*jtot);
    save_binary("atmos_direct", atmos_direct_count.data(), itot*jtot*ktot);
    save_binary("atmos_diffuse", atmos_diffuse_count.data(), itot*jtot*ktot);
}


int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cout << "Add the multiple of 100,000 photons as an argument!" << std::endl;
        return 1;
    }

    const uint64_t n_photons = std::stoi(argv[1]) * 100000;

    run_ray_tracer(n_photons);

    return 0;
}
