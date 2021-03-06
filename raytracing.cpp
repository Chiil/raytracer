#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>


struct Photon;

using Photon_array = Kokkos::View<Photon*, Kokkos::LayoutRight, Kokkos::HostSpace>;
using Array_1d = Kokkos::View<unsigned*, Kokkos::LayoutRight, Kokkos::HostSpace>;
using Array_2d = Kokkos::View<unsigned**, Kokkos::LayoutRight, Kokkos::HostSpace>;


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


struct Initialize_photons
{
    Photon_array photons;
    Kokkos::Random_XorShift64_Pool<> rand_pool;
    const int ktot;
    const double dx;
    const double zenith_angle;

    Initialize_photons(
            Photon_array photons_,
            Kokkos::Random_XorShift64_Pool<> rand_pool_,
            const int ktot_, const double dx_, const double zenith_angle_) :
        photons(photons_), rand_pool(rand_pool_),
        ktot(ktot_), dx(dx_), zenith_angle(zenith_angle_)
    {}

    void operator()(const int n) const
    {
        auto rand_gen = rand_pool.get_state();

        photons(n).position.x = 0.;
        photons(n).position.z = ktot * dx * rand_gen.drand(0., 1.);
        photons(n).direction.x = -std::sin(zenith_angle);
        photons(n).direction.z = -std::cos(zenith_angle);

        rand_pool.free_state(rand_gen);
    }
};


struct Transport_photons
{
    Photon_array photons;
    Kokkos::Random_XorShift64_Pool<> rand_pool;
    const double k_ext;
    const double x_size;
    const double z_size;

    Transport_photons(
            Photon_array photons_,
            Kokkos::Random_XorShift64_Pool<> rand_pool_,
            Array_1d surface_count, Array_1d toa_count,
            const double k_ext_, const double x_size_, const double z_size_) :
        photons(photons_), rand_pool(rand_pool_),
        k_ext(k_ext_), x_size(x_size_), z_size(z_size_)
    {}

    void operator()(const int n) const
    {
        auto rand_gen = rand_pool.get_state();

        const double distance = sample_tau(rand_gen.drand(0., 1.)) / k_ext;
        double dx = photons(n).direction.x * distance;
        double dz = photons(n).direction.z * distance;

        if ((photons(n).position.z + dz) <= 0.)
        {
            const double fac = std::abs(photons(n).position.z / dz);
            dx *= fac;
            dz *= fac;
        }

        photons(n).position.x += dx;
        photons(n).position.z += dz;

        // Cyclic boundary condition in x.
        photons(n).position.x = std::fmod(photons(n).position.x, x_size);

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
    const double zenith_angle;

    Scatter_photons(
            Photon_array photons_, Array_2d atmos_count_,
            Kokkos::Random_XorShift64_Pool<> rand_pool_,
            const double ssa_, const double dx_, const double x_size_, const double zenith_angle_) :
        photons(photons_), atmos_count(atmos_count_),
        rand_pool(rand_pool_),
        ssa(ssa_), dx(dx_), x_size(x_size_), zenith_angle(zenith_angle_)
    {}

    void operator()(const int n) const
    {
        auto rand_gen = rand_pool.get_state();

        const double event = rand_gen.drand(0., 1.);

        if (event >= ssa)
        {
            const int i = photons(n).position.x / dx;
            const int k = photons(n).position.z / dx;

            atmos_count(k, i) += 1;

            photons(n).position.x = 0.;
            photons(n).position.z = x_size * rand_gen.drand(0., 1.);
            photons(n).direction.x = -std::sin(zenith_angle);
            photons(n).direction.z = -std::cos(zenith_angle);
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

    // Array_2d tau("tau", ktot, itot);
    // Array_2d ssa("ssa", ktot, itot);
    // Array_2d g  ("g"  , ktot, itot);

    const double k_ext = 3.e-4;
    const double ssa = 0.5;

    Array_1d surface_count("surface", itot);
    Array_1d toa_count("toa", itot);
    Array_2d atmos_count("atmos", ktot, itot);

    const int n_photons = 1000000;

    const double zenith_angle = 30. * (M_PI/180.);

    const int n_photon_batch = 100;

    Kokkos::Random_XorShift64_Pool<> rand_pool(1);

    Photon_array photons("photons", n_photon_batch);

    Kokkos::parallel_for(
            "Initialize photons",
            n_photon_batch,
            Initialize_photons(
                photons, rand_pool,
                ktot, dx, zenith_angle));

    Kokkos::parallel_for(
            "Transport photons",
            n_photon_batch,
            Transport_photons(
                photons, rand_pool,
                surface_count, toa_count,
                k_ext, x_size, z_size));

    Kokkos::parallel_for(
            "Scatter photons",
            n_photon_batch,
            Scatter_photons(
                photons, atmos_count,
                rand_pool,
                ssa, dx, x_size, zenith_angle));

    std::cout << photons(0).position.x << ", ";
    std::cout << photons(0).position.z << ", ";
    std::cout << photons(0).direction.x << ", ";
    std::cout << photons(0).direction.z << std::endl;
}


int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
   
    run_ray_tracer();

    Kokkos::finalize();

    return 0;
}
