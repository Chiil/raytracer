#include <fstream>
#include <iostream>
#include <iomanip>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>


struct Photon;

using Photon_array = Kokkos::View<Photon*, Kokkos::LayoutRight, Kokkos::HostSpace>;
using Array_1d = Kokkos::View<unsigned int*, Kokkos::LayoutRight, Kokkos::HostSpace>;
using Array_2d = Kokkos::View<unsigned int**, Kokkos::LayoutRight, Kokkos::HostSpace>;


struct Vector
{
    double x;
    double z;
};


struct Photon
{
    Vector position;
    Vector direction;
};


KOKKOS_INLINE_FUNCTION
double sample_tau(const double random_number)
{
    return -1.*log(1.-random_number);
}

KOKKOS_INLINE_FUNCTION
Photon generate_photon(
        const double random_number, const double x_size, const double z_size, const double zenith_angle)
{
    return Photon{ x_size * random_number, z_size, -std::sin(zenith_angle), -std::cos(zenith_angle)};
}

struct Initialize_photons
{
    Photon_array photons;
    Kokkos::Random_XorShift64_Pool<> rand_pool;
    const double x_size;
    const double z_size;
    const double zenith_angle;

    Initialize_photons(
            Photon_array photons_,
            Kokkos::Random_XorShift64_Pool<> rand_pool_,
            const double x_size_, const double z_size_, const double zenith_angle_) :
        photons(photons_), rand_pool(rand_pool_),
        x_size(x_size_), z_size(z_size_), zenith_angle(zenith_angle_)
    {}

    void operator()(const int n) const
    {
        auto rand_gen = rand_pool.get_state();
        photons(n) = generate_photon(rand_gen.drand(0., 1.), x_size, z_size, zenith_angle);
        rand_pool.free_state(rand_gen);
    }
};


struct Transport_photons
{
    Photon_array photons;
    Array_1d surface_count;
    Array_1d toa_count;
    Kokkos::Random_XorShift64_Pool<> rand_pool;
    const double k_ext;
    const double dx_grid;
    const double x_size;
    const double z_size;
    const double zenith_angle;

    Transport_photons(
            Photon_array photons_,
            Array_1d surface_count_, Array_1d toa_count_,
            Kokkos::Random_XorShift64_Pool<> rand_pool_,
            const double k_ext_, const double dx_grid_,
            const double x_size_, const double z_size_, const double zenith_angle_) :
        photons(photons_),
        surface_count(surface_count_), toa_count(toa_count_),
        rand_pool(rand_pool_),
        k_ext(k_ext_), dx_grid(dx_grid_),
        x_size(x_size_), z_size(z_size_), zenith_angle(zenith_angle_)
    {}

    void operator()(const int n) const
    {
        auto rand_gen = rand_pool.get_state();

        while (true)
        {
            const double dn = sample_tau(rand_gen.drand(0., 1.)) / k_ext;
            double dx = photons(n).direction.x * dn;
            double dz = photons(n).direction.z * dn;

            bool surface_exit = false;
            bool toa_exit = false;

            if ((photons(n).position.z + dz) <= 0.)
            {
                const double fac = std::abs(photons(n).position.z / dz);
                dx *= fac;
                dz *= fac;

                surface_exit = true;
            }
            else if ((photons(n).position.z + dz) >= z_size)
            {
                const double fac = std::abs((z_size - photons(n).position.z) / dz);
                dx *= fac;
                dz *= fac;

                toa_exit = true;
            }

            photons(n).position.x += dx;
            photons(n).position.z += dz;

            // Cyclic boundary condition in x.
            photons(n).position.x = std::fmod(photons(n).position.x, x_size);
            if (photons(n).position.x < 0.)
                photons(n).position.x += x_size;

            if (surface_exit || toa_exit)
            {
                const int i = photons(n).position.x / dx_grid;

                // This update is not thread safe.
                if (surface_exit)
                    Kokkos::atomic_increment(&surface_count(i));
                else
                    Kokkos::atomic_increment(&toa_count(i));

                photons(n) = generate_photon(rand_gen.drand(0., 1.), x_size, z_size, zenith_angle);
            }
            else
            {
                break;
            }
        }

        rand_pool.free_state(rand_gen);
    }
};


struct Scatter_photons
{
    Photon_array photons;
    Array_2d atmos_count;
    Kokkos::Random_XorShift64_Pool<> rand_pool;

    const double ssa;
    const double dx;
    const double x_size;
    const double z_size;
    const double zenith_angle;

    Scatter_photons(
            Photon_array photons_, Array_2d atmos_count_,
            Kokkos::Random_XorShift64_Pool<> rand_pool_,
            const double ssa_, const double dx_, const double x_size_, const double z_size_, const double zenith_angle_) :
        photons(photons_), atmos_count(atmos_count_),
        rand_pool(rand_pool_),
        ssa(ssa_), dx(dx_), x_size(x_size_), z_size(z_size_), zenith_angle(zenith_angle_)
    {}

    void operator()(const int n) const
    {
        auto rand_gen = rand_pool.get_state();

        const double event = rand_gen.drand(0., 1.);

        if (event >= ssa)
        {
            const int i = photons(n).position.x / dx;
            const int k = photons(n).position.z / dx;

            Kokkos::atomic_increment(&atmos_count(k, i));

            photons(n) = generate_photon(rand_gen.drand(0., 1.), x_size, z_size, zenith_angle);
        }
        else
        {
            const double angle = rand_gen.drand(0, 2.*M_PI);
            photons(n).direction.x = std::sin(angle);
            photons(n).direction.z = std::cos(angle);
        }

        rand_pool.free_state(rand_gen);
    }
};


void run_ray_tracer()
{
    const double dx = 100.;
    const int itot = 64;
    const int ktot = 64;

    const double x_size = itot*dx;
    const double z_size = ktot*dx;

    const double k_ext = 3.e-4;
    const double ssa = 0.5;

    Array_1d surface_count("surface", itot);
    Array_1d toa_count("toa", itot);
    Array_2d atmos_count("atmos", ktot, itot);

    const double zenith_angle = 30. * (M_PI/180.);

    const int n_photon_batch = 10*1024*1024;

    Kokkos::Random_XorShift64_Pool<> rand_pool(1);

    Photon_array photons("photons", n_photon_batch);

    Kokkos::parallel_for(
            "Initialize photons",
            n_photon_batch,
            Initialize_photons(
                photons, rand_pool,
                x_size, z_size, zenith_angle));

    Kokkos::Timer timer;
    for (int n=0; n<1; ++n)
    {
        Kokkos::parallel_for(
                "Transport photons",
                n_photon_batch,
                Transport_photons(
                    photons,
                    surface_count, toa_count,
                    rand_pool,
                    k_ext, dx,
                    x_size, z_size, zenith_angle));

        Kokkos::parallel_for(
                "Scatter photons",
                n_photon_batch,
                Scatter_photons(
                    photons, atmos_count,
                    rand_pool,
                    ssa, dx, x_size, z_size, zenith_angle));
    }
    Kokkos::fence();

    double duration = timer.seconds();
    std::cout << "Duration: " << std::setprecision(5) << duration << " (s)" << std::endl;

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

    save_binary("surface", surface_count.data(), itot);
    save_binary("toa", toa_count.data(), itot);
    save_binary("atmos", atmos_count.data(), itot*ktot);
}


int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
   
    run_ray_tracer();

    Kokkos::finalize();

    return 0;
}
