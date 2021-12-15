#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>

#include <curand_kernel.h>
#include <float.h>

using Int = unsigned long long;
const Int Atomic_reduce_const = (Int)(-1LL);

// using Int = unsigned int;
// const Int Atomic_reduce_const = (Int)(-1);

// using Float = double;
// const Float Float_epsilon = DBL_EPSILON;
// constexpr int block_size = 768;
// constexpr int grid_size = 64;

using Float = float;
const Float Float_epsilon = FLT_EPSILON;
constexpr int block_size = 512;
constexpr int grid_size = 64;

struct Vector
{
    Float x;
    Float y;
    Float z;

};

static inline __device__
Vector operator*(const Vector v, const Float s) { return Vector{s*v.x, s*v.y, s*v.z}; }
static inline __device__
Vector operator*(const Float s, const Vector v) { return Vector{s*v.x, s*v.y, s*v.z}; }
static inline __device__
Vector operator-(const Vector v1, const Vector v2) { return Vector{v1.x-v2.x, v1.y-v2.y, v1.z-v2.z}; }
static inline __device__
Vector operator+(const Vector v1, const Vector v2) { return Vector{v1.x+v2.x, v1.y+v2.y, v1.z+v2.z}; }


struct Optics_ext
{
    Float gas;
    Float cloud;
};

struct Optics_scat
{
    Float ssa;
    Float asy;
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
Float dot(const Vector v1, const Vector v2)
{
    return v1.x*v2.x + v1.y*v2.y + v1.z*v1.z;
}

__device__
Float norm(const Vector v) { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }


__device__
Vector normalize(const Vector v)
{
    const Float length = norm(v);
    return Vector{ v.x/length, v.y/length, v.z/length};
}

enum class Photon_kind { Direct, Diffuse };

struct Photon
{
    Vector position;
    Vector direction;
    Photon_kind kind;
};


__device__
Float pow2(const Float d) { return d*d; }


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
Float rayleigh(const Float random_number)
{
    const Float q = Float(4.)*random_number - Float(2.);
    const Float d = Float(1.) + pow2(q);
    const Float u = pow(-q + sqrt(d), Float(1./3.));
    return u - Float(1.)/u;
}


__device__
Float henyey(const Float g, const Float random_number)
{
    const Float a = pow2(Float(1.) - pow2(g));
    const Float b = Float(2.)*g*pow2(Float(2.)*random_number*g + Float(1.) - g);
    const Float c = -g/Float(2.) - Float(1.)/(Float(2.)*g);
    return Float(-1.)*(a/b) - c;
}


__device__
Float sample_tau(const Float random_number)
{
    // Prevent log(0) possibility.
    return Float(-1.)*log(-random_number + Float(1.) + Float_epsilon);
}


__device__
inline int float_to_int(const float s_size, const float ds, const int ntot_max)
{
    const int ntot = static_cast<int>(s_size / ds);
    return ntot < ntot_max ? ntot : ntot_max-1;
}

__device__
inline void reset_photon(
        Photon& photon, Int& photons_shot, Int* __restrict__ const toa_down_count,
        const unsigned int random_number_x, const unsigned int random_number_y,
        const Float x_size, const Float y_size, const Float z_size,
        const Float dx_grid, const Float dy_grid, const Float dz_grid,
        const Float dir_x, const Float dir_y, const float dir_z,
        const bool generation_completed,
        const int itot, const int jtot)
{
    if (!generation_completed)
    {
        const int i = random_number_x / static_cast<unsigned int>((1ULL << 32) / itot);
        const int j = random_number_y / static_cast<unsigned int>((1ULL << 32) / jtot);

        photon.position.x = x_size * random_number_x / (1ULL << 32);
        photon.position.y = y_size * random_number_y / (1ULL << 32);
        photon.position.z = z_size;

        photon.direction.x = dir_x;
        photon.direction.y = dir_y;
        photon.direction.z = dir_z;

        photon.kind = Photon_kind::Direct;

        const int ij = i + j*itot;
        atomicAdd(&toa_down_count[ij], 1);
    }
}


template<typename T>
struct Random_number_generator
{
    __device__ Random_number_generator(unsigned int tid)
    {
        curand_init(tid, tid, 0, &state);
    }

    __device__ T operator()();

    curandState state;
};


template<>
__device__ double Random_number_generator<double>::operator()()
{
    return 1. - curand_uniform_double(&state);
}


template<>
__device__ float Random_number_generator<float>::operator()()
{
    return 1.f - curand_uniform(&state);
}


struct Quasi_random_number_generator_2d
{
    __device__ Quasi_random_number_generator_2d(
            curandDirectionVectors32_t* vectors, unsigned int* constants, unsigned int offset)
    {
        curand_init(vectors[0], constants[0], offset, &state_x);
        curand_init(vectors[1], constants[1], offset, &state_y);
    }

    __device__ unsigned int x() { return curand(&state_x); }
    __device__ unsigned int y() { return curand(&state_y); }

    curandStateScrambledSobol32_t state_x;
    curandStateScrambledSobol32_t state_y;
};


__device__
inline void write_photon_out(Int* field_out, Int& photons_shot, const Int inc)
{
    photons_shot += inc;
    atomicAdd(field_out, 1);
}

__global__
void cloud_mask_kernel(
    const Optics_scat* __restrict__ ssa_asy,
    Int* __restrict__ cloud_mask_v,
    Float* __restrict__ cloud_dims,
    const Float dz_grid,
    const int itot, const int jtot, const int ktot)
{
    const int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < ktot)
    {
        cloud_mask_v[k] = 0;
        for (int j=0; j<jtot; ++j)
            for (int i=0; i<jtot; ++i)
            {
                const int ijk = i + j*itot + k*jtot*itot;
                if (ssa_asy[ijk].asy > 0)
                {
                    cloud_mask_v[k] = 1;
                    return;
                }
            }
    }
    __syncthreads();
    if (k==0)
    {
        for (int i=0; i<ktot; ++i)
            if (cloud_mask_v[i]==1)
            {
                cloud_dims[0] = i*dz_grid;
                return;
            }
    }
    if (k==1)
    {
        for (int i=ktot; i>0; --i)
            if (cloud_mask_v[i]==1)
            {
                cloud_dims[1] = (i+1)*dz_grid;
                return;
            }
    }
}

__global__
void ray_tracer_kernel(
        const Int photons_to_shoot,
        Photon* __restrict__ photons,
        Int* __restrict__ toa_down_count,
        Int* __restrict__ toa_up_count,
        Int* __restrict__ surface_down_direct_count,
        Int* __restrict__ surface_down_diffuse_count,
        Int* __restrict__ surface_up_count,
        Int* __restrict__ atmos_direct_count,
        Int* __restrict__ atmos_diffuse_count,
        const Optics_ext* __restrict__ k_ext, const Optics_scat* __restrict__ ssa_asy,
        const Float k_ext_null_cld, const Float k_ext_null_gas,
        const Float surface_albedo,
        const Float x_size, const Float y_size, const Float z_size,
        const Float dx_grid, const Float dy_grid, const Float dz_grid,
        const Float dir_x, const Float dir_y, const Float dir_z,
        const int itot, const int jtot, const int ktot,
        curandDirectionVectors32_t* qrng_vectors, unsigned int* qrng_constants,
        const Float* __restrict__ cloud_dims)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;

    Random_number_generator<Float> rng(n);
    Quasi_random_number_generator_2d qrng(qrng_vectors, qrng_constants, n * photons_to_shoot);

    const Float cloud_min = cloud_dims[0];
    const Float cloud_max = cloud_dims[1];
    const Float s_min = x_size * Float_epsilon;

    // Set up the initial photons.
    const bool completed = false;
    Int photons_shot = 0;

    reset_photon(
            photons[n], photons_shot, toa_down_count,
            qrng.x(), qrng.y(),
            x_size, y_size, z_size,
            dx_grid, dy_grid, dz_grid,
            dir_x, dir_y, dir_z,
            completed,
            itot, jtot);

    Float tau;
    bool surface_exit = false;
    bool toa_exit = false;
    bool transition = false;


    while (photons_shot < photons_to_shoot)
    {

        const bool photon_generation_completed = (photons_shot == photons_to_shoot - 1);
        const bool photon_in_cloud = (photons[n].position.z >= cloud_min && photons[n].position.z <= cloud_max);

        const Float k_ext_null = photon_in_cloud ? k_ext_null_cld : k_ext_null_gas;

        if (!transition)
            tau = sample_tau(rng());

        const Float dn = max(Float_epsilon, sample_tau(rng()) / k_ext_null);
        Float dx = photons[n].direction.x * dn;
        Float dy = photons[n].direction.y * dn;
        Float dz = photons[n].direction.z * dn;

        surface_exit = false;
        toa_exit = false;
        transition = false;

        if (photon_in_cloud)
        {
            const double fac = (photons[n].direction.z > 0) ? (cloud_max-photons[n].position.z) / dz : (cloud_min - photons[n].position.z) / dz;

            if (fac < 1)
            {
                dx *= fac;
                dy *= fac;
                dz *= fac;

                transition = true;

                if ( (photons[n].position.z == cloud_min) && (photons[n].direction.z < 0) )
                    photons[n].position.z -= s_min;

                if ( (photons[n].position.z == cloud_max) && (photons[n].direction.z > 0) )
                    photons[n].position.z += s_min;
            }
        }
        // photon above cloud layer, but about to cross it!
        else if ( (photons[n].position.z > cloud_max) && (photons[n].position.z + dz <= cloud_max) )
        {
            const Float fac = std::abs((photons[n].position.z - cloud_max) / dz);
            dx *= fac;
            dy *= fac;
            dz *= fac;

            transition = true;
        }

        // photon below cloud layer, but about to cross it! (if "constant_gas" is enabled)
        else if (photons[n].position.z < cloud_min && photons[n].position.z + dz >= cloud_min)
        {
            const Float fac = std::abs((photons[n].position.z - cloud_min) / dz);
            dx *= fac;
            dy *= fac;
            dz *= fac;

            transition = true;
        }

        if ((photons[n].position.z + dz) <= Float(0.))
        {
            const Float fac = abs(photons[n].position.z / dz);
            dx *= fac;
            dy *= fac;
            dz *= fac;

            surface_exit = true;
        }
        else if ((photons[n].position.z + dz) >= z_size)
        {
            const Float fac = abs((z_size - photons[n].position.z) / dz);
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
        if (photons[n].position.x < Float(0.))
            photons[n].position.x += x_size;

        // Cyclic boundary condition in y.
        photons[n].position.y = fmod(photons[n].position.y, y_size);
        if (photons[n].position.y < Float(0.))
            photons[n].position.y += y_size;

        // Handle the surface and top exits.
        const int i = float_to_int(photons[n].position.x, dx_grid, itot);
        const int j = float_to_int(photons[n].position.y, dy_grid, jtot);
        const int ij = i + j*itot;

        if (surface_exit)
        {
            if (photons[n].kind == Photon_kind::Direct)
                write_photon_out(&surface_down_direct_count[ij], photons_shot, 1);
            else if (photons[n].kind == Photon_kind::Diffuse)
                write_photon_out(&surface_down_diffuse_count[ij], photons_shot, 1);

            // Surface scatter if smaller than albedo, otherwise absorb
            if (rng() <= surface_albedo)
            {
                write_photon_out(&surface_up_count[ij], photons_shot, Atomic_reduce_const);

                const Float mu_surface = sqrt(rng());
                const Float azimuth_surface = Float(2.*M_PI)*rng();

                photons[n].direction.x = mu_surface*sin(azimuth_surface);
                photons[n].direction.y = mu_surface*cos(azimuth_surface);
                photons[n].direction.z = sqrt(Float(1.) - mu_surface*mu_surface + Float_epsilon);
                photons[n].kind = Photon_kind::Diffuse;
            }
            else
            {
                reset_photon(
                        photons[n], photons_shot, toa_down_count,
                        qrng.x(), qrng.y(),
                        x_size, y_size, z_size,
                        dx_grid, dy_grid, dz_grid,
                        dir_x, dir_y, dir_z,
                        photon_generation_completed,
                        itot, jtot);
            }
        }
        else if (toa_exit)
        {
            write_photon_out(&toa_up_count[ij], photons_shot, 1);

            reset_photon(
                    photons[n], photons_shot, toa_down_count,
                    qrng.x(), qrng.y(),
                    x_size, y_size, z_size,
                    dx_grid, dy_grid, dz_grid,
                    dir_x, dir_y, dir_z,
                    photon_generation_completed,
                    itot, jtot);
        }
        else if (transition)
        {
            tau -= dn * k_ext_null;
        }
        else
        {
            // Calculate the 3D index.
            const int k = float_to_int(photons[n].position.z, dz_grid, ktot);
            const int ijk = i + j*itot + k*itot*jtot;

            // Handle the action.
            const Float random_number = rng();

            // Null collision.
            if (random_number >= ((k_ext[ijk].gas + k_ext[ijk].cloud) / k_ext_null))
            {
            }
            // Scattering.
            else if (random_number <= ssa_asy[ijk].ssa * (k_ext[ijk].gas + k_ext[ijk].cloud) / k_ext_null)
            {
                const bool cloud_scatter = rng() < k_ext[ijk].cloud / (k_ext[ijk].gas + k_ext[ijk].cloud);

                const Float cos_scat = cloud_scatter ? henyey(ssa_asy[ijk].asy, rng()) : rayleigh(rng());
                const Float sin_scat = sqrt(Float(1.) - cos_scat*cos_scat + Float_epsilon);

                Vector t1{Float(0.), Float(0.), Float(0.)};
                if (fabs(photons[n].direction.x) < fabs(photons[n].direction.y))
                {
                    if (fabs(photons[n].direction.x) < fabs(photons[n].direction.z))
                        t1.x = Float(1.);
                    else
                        t1.z = Float(1.);
                }
                else
                {
                    if (fabs(photons[n].direction.y) < fabs(photons[n].direction.z))
                        t1.y = Float(1.);
                    else
                        t1.z = Float(1.);
                }
                t1 = normalize(t1 - photons[n].direction*dot(t1, photons[n].direction));
                Vector t2 = cross(photons[n].direction, t1);

                const Float phi = Float(2.*M_PI)*rng();

                photons[n].direction = cos_scat*photons[n].direction
                        + sin_scat*(sin(phi)*t1 + cos(phi)*t2);

                photons[n].kind = Photon_kind::Diffuse;
            }
            // Absorption.
            else
            {
                if (photons[n].kind == Photon_kind::Direct)
                    write_photon_out(&atmos_direct_count[ijk], photons_shot, 1);
                else
                    write_photon_out(&atmos_diffuse_count[ijk], photons_shot, 1);

                reset_photon(
                        photons[n], photons_shot, toa_down_count,
                        qrng.x(), qrng.y(),
                        x_size, y_size, z_size,
                        dx_grid, dy_grid, dz_grid,
                        dir_x, dir_y, dir_z,
                        photon_generation_completed,
                        itot, jtot);
            }
        }
    }
}
