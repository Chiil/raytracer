#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <string>
#include <time.h>
#include <sys/time.h>
#include <limits>
#include "raytracer_old.h"

//double get_wall_time(){
//    struct timeval time;
//    if (gettimeofday(&time,NULL)){
//        return 0;
//    }
//    return (double)time.tv_sec + (double)time.tv_usec * .000001;
//}
std::random_device rd;
//std::mt19937 mt(rd());
std::uniform_real_distribution<double> dist(0.0, 1.0);

const double eps = std::numeric_limits<double>::epsilon();
const double ds = 1e-3;
const int ZC = 666;
const int SC = 667;
const int AB = 668;
const int BD = 669;

double rayleigh(std::mt19937& mt)
{
    const double q = 4.f*dist(mt)-2.f;
    const double d = 1.f+q*q;
    const double u = pow(-q+sqrt(d), 1.f/3.f);
    return u-1.f/u;
}

double henyey(double g, std::mt19937& mt)
{
    const double a = pow(1.f-pow(g,2.f), 2.f);
    const double b = 2.f*g*pow(2*dist(mt)*g+1.f-g, 2.f);
    const double c = -g/2.f-1.f/(2.f*g);
    return -1.f*(a/b)-c;
}

double sample_tau(std::mt19937& mt)
{
    return -1.f*log(-dist(mt)+1.f);
}

void move_photon(const int ktot, const int itot, const double zsize, const double xsize, const double dx_grid,
                 const double* k_ext, const double* ssa, const double k_null,
                 double* position, double* direction, int& event, std::mt19937& mt)
{
    const double s = sample_tau(mt) / k_null;
    const double s_max = std::min((double(zsize)*(direction[0]>0)-position[0])/direction[0],
                                 (double(xsize)*(direction[1]>0)-position[1])/direction[1]);
    if (s+ds >= s_max)
    {
        position[0] += direction[0]*(s_max+ds);
        position[1] += direction[1]*(s_max+ds);

        if (position[1] >= xsize)
        {
            position[1] = ds;
        }
        else if (position[1] <= 0)
        {
            position[1] = xsize - ds;
        }

        event = BD;
    }
    else
    {
        position[0] += direction[0]*s;
        position[1] += direction[1]*s;
        const double r = dist(mt);
        const int idx = int(position[1]/dx_grid) + int(position[0]/dx_grid) * itot;
        if (r*k_null >= k_ext[idx])
            event = ZC;
        else if (r*k_null <= k_ext[idx]*ssa[idx])
            event = SC;
        else
            event = AB;
    }
}

void hit_event(const int ktot, const int itot, const double zsize, const double xsize, const double dx_grid,
               const int event, const double* k_ext, const double k_ext_gas,
               const double albedo, const double* g,
               double* position, double* direction, bool& f_direct, bool& f_alive, std::mt19937& mt,
               unsigned int* sfc_dir, unsigned int* sfc_dif, unsigned int* atm_cnt)
{
    if (event == SC)
    {
        f_direct = false;
        f_alive = true;
        const int idx = int(position[1]/dx_grid) + int(position[0]/dx_grid) * itot;
        if (dist(mt) < (k_ext[idx]-k_ext_gas)/k_ext[idx]) // cloud scattering
        {
            const double mu_scat  = henyey(g[idx], mt);
            const double angle = acos(std::min(std::max(-1.f+eps,mu_scat),1.f-eps)) * int(-1+2*(dist(mt) > .5f))+ atan2(direction[0], direction[1]);

            direction[0] = sin(angle);
            direction[1] = cos(angle);
        }
        else // gas scattering
        {
            const double mu_scat = rayleigh(mt);
            const double angle= acos(mu_scat) + atan2(direction[0], direction[1]) * int(-1+2*(dist(mt) > .5f));
            direction[0] = sin(angle);
            direction[1] = cos(angle);
        }
    }
    else if (event == AB) // absorption event
    {
        f_alive = false;
        const int idx = int(position[1]/dx_grid) + int(position[0]/dx_grid) * itot;
        atm_cnt[idx] += 1;
    }
    else if (event == ZC) // zero collision event
    {
        f_alive = true;
    }
    else if (event == BD) // boundary hit (surface or top-of-domain
    {
        if (position[0] >= zsize) // left top of domain
        {
            f_alive = false;
        }
        else if (position[0] <= 0.) // surface interaction
        {
            if (f_direct)
                sfc_dir[int(position[1]/dx_grid)] += 1;
            else
                sfc_dif[int(position[1]/dx_grid)] += 1;

            if (dist(mt) > albedo) // absorption by surface
            {
                f_alive = false;
            }
            else // scattering by surface
            {
                f_direct = false;
                position[0] = ds;

                const double mu_sfc = sqrt(dist(mt));
                direction[0] = mu_sfc;
                direction[1] = sin(acos(mu_sfc) * int(-1+2*(dist(mt) > .5f)));

                f_alive = true;
            }
        }
        else
        {
            f_alive = true;
        }

    }
    else
    {
        std::cout<<"oh oh, big problems here"<<std::endl;
    }
}

void trace_ray(const int ktot, const int itot, const double zsize, const double xsize, 
               const double* k_ext, const double* ssa, const double* g,
               const double dx_grid, const double albedo, const double sza_rad,
               const double k_ext_gas, const double k_null,
               const int n_photon, unsigned int* sfc_dir, unsigned int* sfc_dif, unsigned int* atmos_count)
{
    const int n_threads = 4 ;//omp_get_max_threads();
    const int photons_per_block = int(n_photon/n_threads);
    #pragma omp parallel for
    for (int ithread = 0; ithread < n_threads; ++ithread)
    {
        std::mt19937 mt(ithread);
        #pragma omp critical
        std::cout<<"#thread "<<ithread<<std::endl;
        std::vector<unsigned int> sfc_dir_tmp(itot,0);
        std::vector<unsigned int> sfc_dif_tmp(itot,0);
        std::vector<unsigned int> atm_cnt_tmp(itot*ktot,0);

        for (int iphoton = 0; iphoton < photons_per_block; ++iphoton)
        {
            double direction[2] = {-std::cos(sza_rad), -std::sin(sza_rad)};
            double position[2]  = {zsize-ds, dist(mt)*xsize};
            bool f_alive  = true;
            bool f_direct = true;
            int event     = 0;
            while (f_alive)
            {
                move_photon(ktot, itot, zsize, xsize, dx_grid, k_ext, ssa, k_null, position, direction, event, mt);
                hit_event(ktot, itot, zsize, xsize, dx_grid, event, k_ext, k_ext_gas, albedo, g, position, direction, f_direct, f_alive, mt,
                          sfc_dir_tmp.data(), sfc_dif_tmp.data(), atm_cnt_tmp.data());
            }
    
        }
        #pragma omp critical
        for (int ix = 0; ix < itot; ++ix)
        {
            sfc_dir[ix] += sfc_dir_tmp[ix];
            sfc_dif[ix] += sfc_dif_tmp[ix];
        }
        #pragma omp critical
        for (int iz = 0; iz < ktot; ++iz)
            for (int ix = 0; ix < itot; ++ix)
            {
                const int idx = ix+iz*itot;
                atmos_count[idx] += atm_cnt_tmp[idx];
            }
    }
}















