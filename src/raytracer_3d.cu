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


__device__
Vector operator*(const Vector v, const Float s) { return Vector{s*v.x, s*v.y, s*v.z}; }
__device__
Vector operator*(const Float s, const Vector v) { return Vector{s*v.x, s*v.y, s*v.z}; }
__device__
Vector operator-(const Vector v1, const Vector v2) { return Vector{v1.x-v2.x, v1.y-v2.y, v1.z-v2.z}; }
__device__
Vector operator+(const Vector v1, const Vector v2) { return Vector{v1.x+v2.x, v1.y+v2.y, v1.z+v2.z}; }


enum class Photon_kind { Direct, Diffuse };


struct Photon
{
    Vector position;
    Vector direction;
    Photon_kind kind;
};


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
        const Float k_ext_null, const Float k_ext_gas,
        const Float surface_albedo,
        const Float x_size, const Float y_size, const Float z_size,
        const Float dx_grid, const Float dy_grid, const Float dz_grid,
        const Float dir_x, const Float dir_y, const Float dir_z, 
        const int itot, const int jtot, const int ktot,
        curandDirectionVectors32_t* qrng_vectors, unsigned int* qrng_constants)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;

    Random_number_generator<Float> rng(n);
    Quasi_random_number_generator_2d qrng(qrng_vectors, qrng_constants, n * photons_to_shoot);

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
   
    while (photons_shot < photons_to_shoot)
    {        
        const bool photon_generation_completed = (photons_shot == photons_to_shoot - 1);
        const Float dn = max(Float_epsilon, sample_tau(rng()) / k_ext_null);
        Float dx = photons[n].direction.x * dn;
        Float dy = photons[n].direction.y * dn;
        Float dz = photons[n].direction.z * dn;

        bool surface_exit = false;
        bool toa_exit = false;

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


void run_ray_tracer(const Int n_photons)
{
    // Workload per thread
    const Int photons_per_thread = n_photons / (grid_size * block_size);
    std::cout << "Shooting " << n_photons << " photons (" << photons_per_thread << " per thread) " << std::endl;

    //// DEFINE INPUT ////
    // Grid properties.
    const Float dx_grid = 50.;
    const Float dy_grid = 50.;
    const Float dz_grid = 25.;

    const int itot = 128;
    const int jtot = 128;
    const int ktot = 128;

    const Float x_size = itot*dx_grid;
    const Float y_size = jtot*dy_grid;
    const Float z_size = ktot*dz_grid;

    // Radiation properties.
    const Float surface_albedo = 0.2;
    const Float zenith_angle = 50.*(M_PI/180.);
    const Float azimuth_angle = 20.*(M_PI/180.);
    
    const Float dir_x = -std::sin(zenith_angle) * std::cos(azimuth_angle);
    const Float dir_y = -std::sin(zenith_angle) * std::sin(azimuth_angle);
    const Float dir_z = -std::cos(zenith_angle);

    // Input fields.
    const Float k_ext_gas = 1.e-4; // 3.e-4;
    const Float ssa_gas = 0.5;
    const Float asy_gas = 0.;

    const Float k_ext_cloud = 5.e-3;
    const Float ssa_cloud = 0.9;
    const Float asy_cloud = 0.85;

    // Create the spatial fields.
    std::vector<Optics_ext> k_ext(itot*jtot*ktot);
    std::vector<Optics_scat> ssa_asy(itot*jtot*ktot);

    // First add the gases over the entire domain.
    std::fill(k_ext.begin(), k_ext.end(), Optics_ext{k_ext_gas, Float(0.)});
    std::fill(ssa_asy.begin(), ssa_asy.end(), Optics_scat{ssa_gas, asy_gas});

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
                    k_ext[ijk].cloud = k_ext_cloud;
                    ssa_asy[ijk].ssa = (ssa_gas*k_ext_gas + ssa_cloud*k_ext_cloud)
                             / (k_ext_gas + k_ext_cloud);
                    ssa_asy[ijk].asy = (asy_gas*ssa_gas*k_ext_gas + asy_cloud*ssa_cloud*k_ext_cloud)
                             / (ssa_gas*k_ext_gas + ssa_cloud*k_ext_cloud);
                }
            }

    // Set the step size for the transport solver to the maximum extinction coefficient.
    const Float k_ext_null = k_ext_gas + k_ext_cloud;


    //// PREPARE OUTPUT ARRAYS ////
    std::vector<Int> surface_down_direct_count(itot*jtot);
    std::vector<Int> surface_down_diffuse_count(itot*jtot);
    std::vector<Int> surface_up_count(itot*jtot);
    std::vector<Int> toa_down_count(itot*jtot);
    std::vector<Int> toa_up_count(itot*jtot);
    std::vector<Int> atmos_direct_count(itot*jtot*ktot);
    std::vector<Int> atmos_diffuse_count(itot*jtot*ktot);


    //// COPY THE DATA TO THE GPU.
    // Input array.
    Optics_ext* k_ext_gpu = allocate_gpu<Optics_ext>(itot*jtot*ktot);
    Optics_scat* ssa_asy_gpu = allocate_gpu<Optics_scat>(itot*jtot*ktot);

    copy_to_gpu(k_ext_gpu, k_ext.data(), itot*jtot*ktot);
    copy_to_gpu(ssa_asy_gpu, ssa_asy.data(), itot*jtot*ktot);

    // Output arrays. Copy them in order to enable restarts later.
    Int* surface_down_direct_count_gpu = allocate_gpu<Int>(itot*jtot);
    Int* surface_down_diffuse_count_gpu = allocate_gpu<Int>(itot*jtot);
    Int* surface_up_count_gpu = allocate_gpu<Int>(itot*jtot);
    Int* toa_down_count_gpu = allocate_gpu<Int>(itot*jtot);
    Int* toa_up_count_gpu = allocate_gpu<Int>(itot*jtot);
    Int* atmos_direct_count_gpu = allocate_gpu<Int>(itot*jtot*ktot);
    Int* atmos_diffuse_count_gpu = allocate_gpu<Int>(itot*jtot*ktot);

    copy_to_gpu(surface_down_direct_count_gpu, surface_down_direct_count.data(), itot*jtot);
    copy_to_gpu(surface_down_diffuse_count_gpu, surface_down_diffuse_count.data(), itot*jtot);
    copy_to_gpu(surface_up_count_gpu, surface_up_count.data(), itot*jtot);
    copy_to_gpu(toa_down_count_gpu, toa_down_count.data(), itot*jtot);
    copy_to_gpu(toa_up_count_gpu, toa_up_count.data(), itot*jtot);
    copy_to_gpu(atmos_direct_count_gpu, atmos_direct_count.data(), itot*jtot*ktot);
    copy_to_gpu(atmos_diffuse_count_gpu, atmos_diffuse_count.data(), itot*jtot*ktot);


    //// RUN THE RAY TRACER ////
    Photon* photons = allocate_gpu<Photon>(grid_size*block_size);

    curandDirectionVectors32_t* qrng_vectors;
    curandGetDirectionVectors32(
                &qrng_vectors,
                CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6);
    unsigned int* qrng_constants;
    curandGetScrambleConstants32(&qrng_constants);

    curandDirectionVectors32_t* qrng_vectors_gpu = allocate_gpu<curandDirectionVectors32_t>(2);
    copy_to_gpu(qrng_vectors_gpu, qrng_vectors, 2);
    unsigned int* qrng_constants_gpu = allocate_gpu<unsigned int>(2);
    copy_to_gpu(qrng_constants_gpu, qrng_constants, 2);

    dim3 grid{grid_size}, block{block_size};
    
    auto start = std::chrono::high_resolution_clock::now();

    ray_tracer_kernel<<<grid, block>>>(
            photons_per_thread, photons,
            toa_down_count_gpu, toa_up_count_gpu,
            surface_down_direct_count_gpu, surface_down_diffuse_count_gpu, surface_up_count_gpu,
            atmos_direct_count_gpu, atmos_diffuse_count_gpu,
            k_ext_gpu, ssa_asy_gpu,
            k_ext_null, k_ext_gas, surface_albedo,
            x_size, y_size, z_size,
            dx_grid, dy_grid, dz_grid,
            dir_x, dir_y, dir_z,
            itot, jtot, ktot,
            qrng_vectors_gpu, qrng_constants_gpu);

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
            binary_file.write(reinterpret_cast<const char*>(ptr), size*sizeof(Int));
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
        std::cout << "The number of photons is must be a power of two (2**n), please add the exponent n" << std::endl;
        return 1;
    }

    const Int n_photons = std::pow(Int(2), static_cast<Int>(std::stoi(argv[1])));
    
    if (n_photons < grid_size * block_size)
    {
        std::cerr << "Sorry, the number of photons must be larger than " << grid_size * block_size
            << " (n >= " << std::log2(grid_size*block_size) << ") to guarantee one photon per thread" << std::endl;
        return 1;
    }
    
    run_ray_tracer(n_photons);

    return 0;
}
