#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>

#include <curand_kernel.h>
#include <float.h>


#define uint64_t unsigned long long


struct Vector
{
    double x;
    double y;
    double z;
};


__device__
Vector cross(const Vector v1, const Vector v2)
{
    return Vector{
            v1.y*v2.z - v1.z*v2.y,
            v1.z*v2.x - v1.x*v2.z,
            v1.x*v2.y - v1.y*v2.x};
}


__device__
double dot(const Vector v1, const Vector v2)
{
    return v1.x*v2.x + v1.y*v2.y + v1.z*v1.z;
}


__device__
double norm(const Vector v) { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }


__device__
Vector normalize(const Vector v)
{
    const double length = norm(v);
    return Vector{ v.x/length, v.y/length, v.z/length};
}


__device__
Vector operator*(const Vector v, const double s) { return Vector{s*v.x, s*v.y, s*v.z}; }
__device__
Vector operator*(const double s, const Vector v) { return Vector{s*v.x, s*v.y, s*v.z}; }
__device__
Vector operator-(const Vector v1, const Vector v2) { return Vector{v1.x-v2.x, v1.y-v2.y, v1.z-v2.z}; }
__device__
Vector operator+(const Vector v1, const Vector v2) { return Vector{v1.x+v2.x, v1.y+v2.y, v1.z+v2.z}; }


enum class Photon_kind { Direct, Diffuse };
enum class Photon_status { Enabled, Disabled };


struct Photon
{
    Vector position;
    Vector direction;
    Photon_kind kind;
    Photon_status status;
};


__device__
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


__device__
double rayleigh(const double random_number)
{
    const double q = 4.*random_number - 2.;
    const double d = 1. + pow2(q);
    const double u = pow(-q + sqrt(d), 1./3.);
    return u - 1./u;
}


__device__
double henyey(const double g, const double random_number)
{
    const double a = pow2(1. - pow2(g));
    const double b = 2.*g*pow2(2.*random_number*g + 1. - g);
    const double c = -g/2. - 1./(2.*g);
    return -1.*(a/b) - c;
}


__device__
double sample_tau(const double random_number)
{
    // return -1.*log(-random_number + 1.) + std::numeric_limits<double>::epsilon();
    return -1.*log(-random_number + 1.) + DBL_EPSILON;
}


__device__
void reset_photon(
        Photon& photon,
        const double random_number_x, const double random_number_y,
        const double x_size, const double y_size, const double z_size,
        const double zenith_angle, const double azimuth_angle)
{
    photon.position.x = x_size * random_number_x;
    photon.position.y = y_size * random_number_y;
    photon.position.z = z_size;
    photon.direction.x = -sin(zenith_angle) * cos(azimuth_angle);
    photon.direction.y = -sin(zenith_angle) * sin(azimuth_angle);
    photon.direction.z = -cos(zenith_angle);
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
    const int n = threadIdx.x;

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


__device__
void ray_tracer_step1_kernel(
        Photon* __restrict__ photons, const int rng_offset,
        const bool photon_generation_completed,
        uint64_t& n_photons_in, uint64_t& n_photons_out,
        uint64_t* __restrict__ toa_down_count,
        uint64_t* __restrict__ toa_up_count,
        uint64_t* __restrict__ surface_down_direct_count,
        uint64_t* __restrict__ surface_down_diffuse_count,
        uint64_t* __restrict__ surface_up_count,
        const double k_ext_null,
        const double surface_albedo,
        double x_size, double y_size, double z_size,
        double dx_grid, double dy_grid, double dz_grid,
        double zenith_angle, double azimuth_angle, 
        const int itot)
{
    const int n = threadIdx.x;

//    const bool photon_generation_completed = false;

    Random_number_generator<double> rng(n + n_photons_in + rng_offset);

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
            const double fac = abs(photons[n].position.z / dz);
            dx *= fac;
            dy *= fac;
            dz *= fac;

            surface_exit = true;
        }
        else if ((photons[n].position.z + dz) >= z_size)
        {
            const double fac = abs((z_size - photons[n].position.z) / dz);
            dx *= fac;
            dy *= fac;
            dz *= fac;

            toa_exit = true;
        }

        photons[n].position.x += dx;
        photons[n].position.y += dy;
        photons[n].position.z += dz;

        // Cyclic boundary condition in x.
        photons[n].position.x = fmod(photons[n].position.x, x_size);
        if (photons[n].position.x < 0.)
            photons[n].position.x += x_size;

        // Cyclic boundary condition in y.
        photons[n].position.y = fmod(photons[n].position.y, y_size);
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
                n_photons_out += 1;//atomicAdd(&n_photons_out[n], 1);
                atomicAdd(&surface_down_direct_count[ij], 1);
            }
            else if (photons[n].kind == Photon_kind::Diffuse
                    && photons[n].status == Photon_status::Enabled)
            {
                n_photons_out += 1;//atomicAdd(&n_photons_out[n], 1);
                atomicAdd(&surface_down_diffuse_count[ij], 1);
            }

            // Surface scatter if smaller than albedo, otherwise absorb
            if (rng() <= surface_albedo)
            {
                if (photons[n].status == Photon_status::Enabled)
                {
                    // Adding 0xffffffffffffffffULL is equal to subtracting one.
                    n_photons_out += 0xffffffffffffffffULL; //atomicAdd(&n_photons_out[n], 0xffffffffffffffffULL);
                    atomicAdd(&surface_up_count[ij], 1);
                }

                const double mu_surface = sqrt(rng());
                const double azimuth_surface = 2.*M_PI*rng();
                // CvH: is this correct?
                photons[n].direction.x = mu_surface*sin(azimuth_surface);
                photons[n].direction.y = mu_surface*cos(azimuth_surface);
                photons[n].direction.z = sqrt(1. - mu_surface*mu_surface);
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
                    n_photons_in += 1;
                    //atomicAdd(&n_photons_in[n], 1);

                    const int i_new = photons[n].position.x / dx_grid;
                    const int j_new = photons[n].position.y / dy_grid;
                    const int ij_new = i_new + j_new*itot;

                    atomicAdd(&toa_down_count[ij_new], 1);
                }
            }
        }
        else if (toa_exit)
        {
            if (photons[n].status == Photon_status::Enabled)
            {
                n_photons_out += 1; //atomicAdd(&n_photons_out[n], 1);
                atomicAdd(&toa_up_count[ij], 1);
            }

            reset_photon(
                    photons[n], rng(), rng(),
                    x_size, y_size, z_size,
                    zenith_angle, azimuth_angle);


            if (photon_generation_completed)
                photons[n].status = Photon_status::Disabled;

            if (photons[n].status == Photon_status::Enabled)
            {
                n_photons_in += 1;//atomicAdd(&n_photons_in[n], 1);
    

                const int i_new = photons[n].position.x / dx_grid;
                const int j_new = photons[n].position.y / dy_grid;
                const int ij_new = i_new + j_new*itot;

                atomicAdd(&toa_down_count[ij_new], 1);
            }
        }
        else
        {
            break;
        }
    }
}


__device__
void ray_tracer_step2_kernel(
        Photon* __restrict__ photons, const int rng_offset,
        const bool photon_generation_completed,
        uint64_t& n_photons_in, uint64_t& n_photons_out,
        uint64_t* __restrict__ toa_down_count,
        uint64_t* __restrict__ toa_up_count,
        uint64_t* __restrict__ surface_down_direct_count,
        uint64_t* __restrict__ surface_down_diffuse_count,
        uint64_t* __restrict__ surface_up_count,
        uint64_t* __restrict__ atmos_direct_count,
        uint64_t* __restrict__ atmos_diffuse_count,
        double* __restrict__ k_ext, double* __restrict__ ssa, double* __restrict__ asy,
        const double k_ext_null, const double k_ext_gas,
        const double surface_albedo,
        double x_size, double y_size, double z_size,
        double dx_grid, double dy_grid, double dz_grid,
        double zenith_angle, double azimuth_angle, 
        const int itot, const int jtot)
{
    const int n =  threadIdx.x;

    Random_number_generator<double> rng(n + n_photons_in + rng_offset);

    const int i = photons[n].position.x / dx_grid;
    const int j = photons[n].position.y / dy_grid;
    const int k = photons[n].position.z / dz_grid;

    const int ijk = i + j*itot + k*itot*jtot;

    const double random_number = rng();

    // Null collision.
    if (random_number >= (k_ext[ijk] / k_ext_null))
    {
    }
    // Scattering.
    else if (random_number <= ssa[ijk] * k_ext[ijk] / k_ext_null)
    {
        const bool cloud_scatter = rng() < (k_ext[ijk] - k_ext_gas) / k_ext[ijk];
        const double cos_scat = cloud_scatter ? henyey(asy[ijk], rng()) : rayleigh(rng());
        const double sin_scat = sqrt(1. - cos_scat*cos_scat);

        Vector t1{0., 0., 0.};
        if (fabs(photons[n].direction.x) < fabs(photons[n].direction.y))
        {
            if (fabs(photons[n].direction.x) < fabs(photons[n].direction.z))
                t1.x = 1;
            else
                t1.z = 1;
        }
        else
        {
            if (fabs(photons[n].direction.y) < fabs(photons[n].direction.z))
                t1.y = 1;
            else
                t1.z = 1;
        }
        t1 = normalize(t1 - photons[n].direction*dot(t1, photons[n].direction));
        Vector t2 = cross(photons[n].direction, t1);

        const double phi = 2.*M_PI*rng();

        photons[n].direction = cos_scat*photons[n].direction
                + sin_scat*(sin(phi)*t1 + cos(phi)*t2);

        photons[n].kind = Photon_kind::Diffuse;
    }
    // Absorption.
    else
    {
        if (photons[n].status == Photon_status::Enabled)
        {
            n_photons_out += 1;//atomicAdd(&n_photons_out[n], 1);

            if (photons[n].kind == Photon_kind::Direct)
                atomicAdd(&atmos_direct_count[ijk], 1);
            else
                atomicAdd(&atmos_diffuse_count[ijk], 1);
        }

        reset_photon(
                photons[n], rng(), rng(),
                x_size, y_size, z_size,
                zenith_angle, azimuth_angle);

        if (photon_generation_completed)
            photons[n].status = Photon_status::Disabled;

        if (photons[n].status == Photon_status::Enabled)
        {
            n_photons_in += 1;// atomicAdd(&n_photons_in[n], 1);

            const int i_new = photons[n].position.x / dx_grid;
            const int j_new = photons[n].position.y / dy_grid;
            const int ij_new = i_new + j_new*itot;

            atomicAdd(&toa_down_count[ij_new], 1);
        }
    }
}

__global__
void ray_tracer_kernel(
        const int photons_to_shoot, const int n_iter,
        Photon* __restrict__ photons,
        uint64_t* __restrict__ n_photons_in, uint64_t* __restrict__ n_photons_out,
        uint64_t* __restrict__ toa_down_count,
        uint64_t* __restrict__ toa_up_count,
        uint64_t* __restrict__ surface_down_direct_count,
        uint64_t* __restrict__ surface_down_diffuse_count,
        uint64_t* __restrict__ surface_up_count,
        uint64_t* __restrict__ atmos_direct_count,
        uint64_t* __restrict__ atmos_diffuse_count,
        double* __restrict__ k_ext, double* __restrict__ ssa, double* __restrict__ asy,
        const double k_ext_null, const double k_ext_gas,
        const double surface_albedo,
        double x_size, double y_size, double z_size,
        double dx_grid, double dy_grid, double dz_grid,
        double zenith_angle, double azimuth_angle, 
        const int itot, const int jtot)
{
    //uint64_t n_photons_in_loc = n_photons_in;
    uint64_t photons_to_shoot_loc = photons_to_shoot;
    //uint64_t photons_to_shoot_loc = photons_to_shoot;
    bool photon_generation_completed = false;
    
    uint64_t tot_photon_in = blockDim.x;
    uint64_t tot_photon_out = 0;
    const int n = threadIdx.x;
    
    while ((tot_photon_in<photons_to_shoot_loc) || ( tot_photon_in > tot_photon_out))
    {
        for (int i=0; i<n_iter; ++i)
        {
            ray_tracer_step1_kernel(
                photons, i,
                photon_generation_completed,
                n_photons_in[n], n_photons_out[n],
                toa_down_count, toa_up_count,
                surface_down_direct_count, surface_down_diffuse_count, surface_up_count,
                k_ext_null, surface_albedo,
                x_size, y_size, z_size,
                dx_grid, dy_grid, dz_grid,
                zenith_angle, azimuth_angle,
                itot);

            ray_tracer_step2_kernel(
                 photons, i,
                 photon_generation_completed,
                 n_photons_in[n], n_photons_out[n],
                 toa_down_count, toa_up_count,
                 surface_down_direct_count, surface_down_diffuse_count, surface_up_count,
                 atmos_direct_count, atmos_diffuse_count,
                 k_ext, ssa, asy,
                 k_ext_null, k_ext_gas, surface_albedo,
                 x_size, y_size, z_size,
                 dx_grid, dy_grid, dz_grid,
                 zenith_angle, azimuth_angle,
                 itot, jtot);
        
        }
        __syncthreads();
        tot_photon_in = 0;
        tot_photon_out = 0;
        
        for (int nt=0; nt<blockDim.x; ++nt)
        {
            tot_photon_in += n_photons_in[nt];
            tot_photon_out += n_photons_out[nt];
        }
        photon_generation_completed = tot_photon_in >= photons_to_shoot_loc;
        //printf("%lu %lu %d\n",tot_photon_in, tot_photon_out,photon_generation_completed);
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
    constexpr int n_iter = 1 << 8;
    constexpr int block_size = 128;
    Photon* photons = allocate_gpu<Photon>(block_size);

    std::vector<uint64_t> n_photons_in(block_size, 1);
    std::vector<uint64_t> n_photons_out(block_size, 0);

    uint64_t* n_photons_in_gpu = allocate_gpu<uint64_t>(block_size);
    uint64_t* n_photons_out_gpu = allocate_gpu<uint64_t>(block_size);

    copy_to_gpu(n_photons_in_gpu, n_photons_in.data(), block_size);
    copy_to_gpu(n_photons_out_gpu, n_photons_out.data(), block_size);

    dim3 grid{1}, block{block_size};
    //dim3 grid{n_photons_batch/block_size}, block{block_size};
    
    auto start = std::chrono::high_resolution_clock::now();

    ray_tracer_init_kernel<<<grid, block>>>(
            photons,
            toa_down_count_gpu,
            x_size, y_size, z_size,
            dx_grid, dy_grid, dz_grid,
            zenith_angle, azimuth_angle,
            itot);
    
    ray_tracer_kernel<<<grid, block>>>(
            n_photons, n_iter, photons,
            n_photons_in_gpu, n_photons_out_gpu,
            toa_down_count_gpu, toa_up_count_gpu,
            surface_down_direct_count_gpu, surface_down_diffuse_count_gpu, surface_up_count_gpu,
            atmos_direct_count_gpu, atmos_diffuse_count_gpu,
            k_ext_gpu, ssa_gpu, asy_gpu,
            k_ext_null, k_ext_gas, surface_albedo,
            x_size, y_size, z_size,
            dx_grid, dy_grid, dz_grid,
            zenith_angle, azimuth_angle,
            itot, jtot);

    cuda_safe_call(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "Duration: " << std::setprecision(5) << duration << " (s)" << std::endl;
    //// END RUNNING OF RAY TRACER ////


    //// COPY OUTPUT BACK TO CPU ////
    copy_from_gpu(n_photons_in.data(), n_photons_in_gpu, block_size);
    copy_from_gpu(n_photons_out.data(), n_photons_out_gpu, block_size);

    copy_from_gpu(surface_down_direct_count.data(), surface_down_direct_count_gpu, itot*jtot);
    copy_from_gpu(surface_down_diffuse_count.data(), surface_down_diffuse_count_gpu, itot*jtot);
    copy_from_gpu(surface_up_count.data(), surface_up_count_gpu, itot*jtot);
    copy_from_gpu(toa_down_count.data(), toa_down_count_gpu, itot*jtot);
    copy_from_gpu(toa_up_count.data(), toa_up_count_gpu, itot*jtot);
    copy_from_gpu(atmos_direct_count.data(), atmos_direct_count_gpu, itot*jtot*ktot);
    copy_from_gpu(atmos_diffuse_count.data(), atmos_diffuse_count_gpu, itot*jtot*ktot);

    uint64_t toa_down = 0;
    uint64_t photons_in = 0;
    uint64_t photons_out = 0;
    for (int j=0; j<jtot; ++j)
        for (int i=0; i<itot; ++i)
            toa_down += toa_down_count[i+j*itot];
    
    for (int n=0; n<block_size; ++n)
    {
        photons_in += n_photons_in[n];
        photons_out += n_photons_out[n];
    }
    std::cout << "CvH: " << photons_in << ", " << photons_out << ", " << toa_down << std::endl;


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
