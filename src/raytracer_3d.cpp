#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif


inline uint64_t rotl(const uint64_t x, int k)
{
	return (x << k) | (x >> (64 - k));
}


// This is the xoroshiro128+ generator of Blackman and Vigna.
// It has a uint64_t[2] as state.
inline uint64_t next_xoroshiro_128_plus(uint64_t* __restrict__ s)
{
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    s[1] = rotl(s1, 37);

    return result;
}


// This is SplitMix64, a separate RNG to inialize the xo_ RNG.
// It has a single uint64_t as state.
inline uint64_t next_sr64(uint64_t& s)
{
	uint64_t z = (s += 0x9e3779b97f4a7c15);

	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb;

	return z ^ (z >> 31);
}


class Random_number_generator
{
    public:
        Random_number_generator(const int seed)
        {
            uint64_t init_state = seed;

            state[0] = next_sr64(init_state);
            state[1] = next_sr64(init_state);
        };

        inline double fp64() { return (next_xoroshiro_128_plus(state) >> 11) * 0x1.0p-53; }
        template<typename T> inline int sign() { return static_cast<T>(-1 + 2*int(next_xoroshiro_128_plus(state) >> 63)); }

    private:
        uint64_t state[2];
};


struct Vector
{
    double x;
    double y;
    double z;
};


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


void run_ray_tracer(const uint64_t n_photons)
{
    //// DEFINE INPUT ////
    // Grid properties.
    const double dx_grid = 50.;
    const int itot = 128;
    const int jtot = 128;
    const int ktot = 64;

    const double x_size = itot*dx_grid;
    const double y_size = jtot*dx_grid;
    const double z_size = ktot*dx_grid;

    // Radiation properties.
    const double surface_albedo = 0.2;
    const double zenith_angle = 40.*(M_PI/180.);
    const double azimuth_angle = 20.*(M_PI/180.);
    constexpr int n_photons_batch = 1 << 16;

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
                if (  i*dx_grid > 2700. && i*dx_grid < 3700.
                   && j*dx_grid > 2700. && k*dx_grid < 3700.
                   && k*dx_grid > 1000. && k*dx_grid < 1500.)
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
    std::vector<uint64_t> atmos_count(itot*jtot*ktot);


    //// RUN THE RAY TRACER ////
    std::vector<Photon> photons(n_photons_batch);

    auto start = std::chrono::high_resolution_clock::now();

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
            const int j = photons[n].position.y / dx_grid;
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
                    double dz = photons[n].direction.z * dn;

                    bool surface_exit = false;
                    bool toa_exit = false;

                    if ((photons[n].position.z + dz) <= 0.)
                    {
                        const double fac = std::abs(photons[n].position.z / dz);
                        dx *= fac;
                        dz *= fac;

                        surface_exit = true;
                    }
                    else if ((photons[n].position.z + dz) >= z_size)
                    {
                        const double fac = std::abs((z_size - photons[n].position.z) / dz);
                        dx *= fac;
                        dz *= fac;

                        toa_exit = true;
                    }

                    photons[n].position.x += dx;
                    photons[n].position.z += dz;

                    // Cyclic boundary condition in x.
                    photons[n].position.x = std::fmod(photons[n].position.x, x_size);
                    if (photons[n].position.x < 0.)
                        photons[n].position.x += x_size;

                    // Handle the surface and top exits.
                    const int i = photons[n].position.x / dx_grid;
                    const int j = photons[n].position.y / dx_grid;
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

                        // Scatter if smaller than albedo, otherwise absorb
                        if (rg.fp64() <= surface_albedo)
                        {
                            if (photons[n].status == Photon_status::Enabled)
                            {
                                --n_photons_out;

                                #pragma omp atomic
                                ++surface_up_count[ij];
                            }

                            const double mu_surface = std::sqrt(rg.fp64());
                            photons[n].direction.x = mu_surface*rg.sign<double>();
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
                                const int j_new = photons[n].position.y / dx_grid;
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
                            const int j_new = photons[n].position.y / dx_grid;
                            #pragma omp atomic
                            ++toa_down_count[i_new + j_new*itot];
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
                const int j = photons[n].position.y / dx_grid;
                const int k = photons[n].position.z / dx_grid;

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
                    const double mu_scat = cloud_scatter ? henyey(asy[ijk], rg.fp64()) : rayleigh(rg.fp64());
                    const double angle = rg.sign<double>() * std::acos(mu_scat)
                        + std::atan2(photons[n].direction.x, photons[n].direction.z);

                    photons[n].direction.x = std::sin(angle);
                    photons[n].direction.z = std::cos(angle);
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
                        const int j_new = photons[n].position.y / dx_grid;
                        const int ij_new = i_new + j_new*itot;
                        #pragma omp atomic
                        ++toa_down_count[ij_new];
                    }
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "Duration: " << std::setprecision(5) << duration << " (s)" << std::endl;
    //// END RUNNING OF RAY TRACER ////


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
    save_binary("atmos", atmos_count.data(), itot*jtot*ktot);
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
