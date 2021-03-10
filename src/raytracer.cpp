#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif


struct Vector
{
    double x;
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
    return -1.*std::log(-random_number + 1.);
}


void reset_photon(
        Photon& photon,
        const double random_number, const double x_size, const double z_size, const double zenith_angle)
{
    photon.position.x = x_size * random_number;
    photon.position.z = z_size;
    photon.direction.x = -std::sin(zenith_angle);
    photon.direction.z = -std::cos(zenith_angle);
    photon.kind = Photon_kind::Direct;
    photon.status = Photon_status::Enabled;
}


void run_ray_tracer(const int n_photons)
{
    //// DEFINE INPUT ////
    // Grid properties.
    const double dx_grid = 25.;
    const int itot = 256;
    const int ktot = 128;

    const double x_size = itot*dx_grid;
    const double z_size = ktot*dx_grid;

    // Radiation properties.
    const double surface_albedo = 0.2;
    const double zenith_angle = 50.*(M_PI/180.);
    constexpr int n_photons_batch = 1 << 16;

    // Input fields.
    const double k_ext_gas = 1.e-4; // 3.e-4;
    const double ssa_gas = 0.5;
    const double asy_gas = 0.;

    const double k_ext_cloud = 5.e-3;
    const double ssa_cloud = 0.9;
    const double asy_cloud = 0.85;

    // Create the spatial fields.
    std::vector<double> k_ext(itot*ktot);
    std::vector<double> ssa(itot*ktot);
    std::vector<double> asy(itot*ktot);

    // First add the gases over the entire domain.
    std::fill(k_ext.begin(), k_ext.end(), k_ext_gas);
    std::fill(ssa.begin(), ssa.end(), ssa_gas);

    // Add a block cloud.
    for (int k=0; k<ktot; ++k)
        for (int i=0; i<itot; ++i)
        {
            if (i*dx_grid > 4000 && i*dx_grid < 5000 && k*dx_grid > 1000 && k*dx_grid < 1500.)
            {
                k_ext[i + k*itot] = k_ext_gas + k_ext_cloud;
                ssa[i + k*itot] = (ssa_gas*k_ext_gas + ssa_cloud*k_ext_cloud)
                                / (k_ext_gas + k_ext_cloud);
                asy[i + k*itot] = (asy_gas*ssa_gas*k_ext_gas + asy_cloud*ssa_cloud*k_ext_cloud)
                                / (ssa_gas*k_ext_gas + ssa_cloud*k_ext_cloud);
            }
        }

    // Set the step size for the transport solver to the maximum extinction coefficient.
    const double k_ext_null = *std::max_element(k_ext.begin(), k_ext.end());


    //// PREPARE OUTPUT ARRAYS ////
    std::vector<unsigned int> surface_down_direct_count(itot);
    std::vector<unsigned int> surface_down_diffuse_count(itot);
    std::vector<unsigned int> surface_up_count(itot);
    std::vector<unsigned int> toa_down_count(itot);
    std::vector<unsigned int> toa_up_count(itot);
    std::vector<unsigned int> atmos_count(itot*ktot);


    //// RUN THE RAY TRACER ////
    std::vector<Photon> photons(n_photons_batch);

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int n=0; n<n_photons_batch; ++n)
    {
        #ifdef _OPENMP
        const int seed = omp_get_thread_num();
        #else
        const int seed = 0;
        #endif

        thread_local std::mt19937_64 mt(seed);
        std::uniform_real_distribution<double> dist(0., 1.);

        reset_photon(
                photons[n],
                dist(mt), x_size, z_size, zenith_angle);

        const int i = photons[n].position.x / dx_grid;
        #pragma omp atomic
        ++toa_down_count[i];
    }

    int n_photons_in = n_photons_batch;
    int n_photons_out = 0;

    while ( (n_photons_in < n_photons) || (n_photons_in > n_photons_out))
    {
        const bool photon_generation_completed = n_photons_in >= n_photons;

        // Transport the photons
        #pragma omp parallel for reduction(+:n_photons_in) reduction(+:n_photons_out)
        for (int n=0; n<n_photons_batch; ++n)
        {
            #ifdef _OPENMP
            const int seed = omp_get_thread_num();
            #else
            const int seed = 0;
            #endif

            thread_local std::mt19937_64 mt(seed);
            std::uniform_real_distribution<double> dist(0., 1.);

            while (true)
            {
                const double dn = sample_tau(dist(mt)) / k_ext_null;
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

                if (surface_exit)
                {
                    if (photons[n].kind == Photon_kind::Direct
                            && photons[n].status == Photon_status::Enabled)
                    {
                        ++n_photons_out;

                        #pragma omp atomic
                        ++surface_down_direct_count[i];
                    }
                    else if (photons[n].kind == Photon_kind::Diffuse
                            && photons[n].status == Photon_status::Enabled)
                    {
                        ++n_photons_out;

                        #pragma omp atomic
                        ++surface_down_diffuse_count[i];
                    }

                    // Scatter if smaller than albedo, otherwise absorb
                    if (dist(mt) <= surface_albedo)
                    {
                        if (photons[n].status == Photon_status::Enabled)
                        {
                            --n_photons_out;
                            #pragma omp atomic
                            ++surface_up_count[i];
                        }

                        const double mu_surface = sqrt(dist(mt));
                        photons[n].direction.x = mu_surface;
                        photons[n].direction.z = std::sin(std::acos(mu_surface) * int(-1.+2.*(dist(mt) > .5)));
                        photons[n].kind = Photon_kind::Diffuse;
                    }
                    else
                    {
                        reset_photon(photons[n], dist(mt), x_size, z_size, zenith_angle);

                        if (photon_generation_completed)
                            photons[n].status = Photon_status::Disabled;

                        if (photons[n].status == Photon_status::Enabled)
                        {
                            ++n_photons_in;

                            const int i_new = photons[n].position.x / dx_grid;
                            #pragma omp atomic
                            ++toa_down_count[i_new];
                        }
                    }
                }
                else if (toa_exit)
                {
                    if (photons[n].status == Photon_status::Enabled)
                    {
                        ++n_photons_out;

                        #pragma omp atomic
                        ++toa_up_count[i];
                    }

                    reset_photon(photons[n], dist(mt), x_size, z_size, zenith_angle);

                    if (photon_generation_completed)
                        photons[n].status = Photon_status::Disabled;

                    if (photons[n].status == Photon_status::Enabled)
                    {
                        ++n_photons_in;

                        const int i_new = photons[n].position.x / dx_grid;
                        #pragma omp atomic
                        ++toa_down_count[i_new];
                    }
                }
                else
                {
                    break;
                }
            }
        }

        // Handle the collision events.
        #pragma omp parallel for reduction(+:n_photons_in) reduction(+:n_photons_out)
        for (int n=0; n<n_photons_batch; ++n)
        {
            #ifdef _OPENMP
            const int seed = omp_get_thread_num();
            #else
            const int seed = 0;
            #endif

            thread_local std::mt19937_64 mt(seed);
            std::uniform_real_distribution<double> dist(0., 1.);

            const int i = photons[n].position.x / dx_grid;
            const int k = photons[n].position.z / dx_grid;

            const double random_number = dist(mt);

            // Null collision.
            if (random_number >= (k_ext[i + k*itot] / k_ext_null))
            {
            }
            // Scattering.
            else if (random_number <= ssa[i + k*itot] * k_ext[i + k*itot] / k_ext_null)
            {
                const bool cloud_scatter = dist(mt) < (k_ext[i + k*itot] - k_ext_gas) / k_ext[i + k*itot];
                const double mu_scat = cloud_scatter ? henyey(asy[i + k*itot], dist(mt)) : rayleigh(dist(mt));
                const double angle = (-1.+2.*(dist(mt)>.5)) * std::acos(mu_scat)
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
                    ++atmos_count[i + k*itot];
                }

                reset_photon(photons[n], dist(mt), x_size, z_size, zenith_angle);

                if (photon_generation_completed)
                    photons[n].status = Photon_status::Disabled;

                if (photons[n].status == Photon_status::Enabled)
                {
                    ++n_photons_in;

                    const int i_new = photons[n].position.x / dx_grid;
                    #pragma omp atomic
                    ++toa_down_count[i_new];
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
            binary_file.write(reinterpret_cast<const char*>(ptr), size*sizeof(unsigned int));
        else
        {
            std::string error = "Cannot write file \"" + name + ".bin\"";
            throw std::runtime_error(error);
        }
    };

    save_binary("toa_down", toa_down_count.data(), itot);
    save_binary("toa_up", toa_up_count.data(), itot);
    save_binary("surface_down_direct", surface_down_direct_count.data(), itot);
    save_binary("surface_down_diffuse", surface_down_diffuse_count.data(), itot);
    save_binary("surface_up", surface_up_count.data(), itot);
    save_binary("atmos", atmos_count.data(), itot*ktot);
}


int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cout << "Add the multiple of 100,000 photons as an argument!" << std::endl;
        return 1;
    }

    const int n_photons = std::stoi(argv[1]) * 100000;

    run_ray_tracer(n_photons);

    return 0;
}
