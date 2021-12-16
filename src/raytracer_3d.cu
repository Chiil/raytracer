#include <raytracer_kernels.cu>

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
                if (  i*dx_grid >= 0. && i*dx_grid < 800.
                   && j*dy_grid >= 0. && j*dy_grid < 800.
                   && k*dz_grid >= 800. && k*dz_grid < 1200.)
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
    std::vector<Float> surface_down_direct_count(itot*jtot);
    std::vector<Float> surface_down_diffuse_count(itot*jtot);
    std::vector<Float> surface_up_count(itot*jtot);
    std::vector<Float> toa_down_count(itot*jtot);
    std::vector<Float> toa_up_count(itot*jtot);
    std::vector<Float> atmos_direct_count(itot*jtot*ktot);
    std::vector<Float> atmos_diffuse_count(itot*jtot*ktot);


    //// COPY THE DATA TO THE GPU.
    // Input array.
    Optics_ext* k_ext_gpu = allocate_gpu<Optics_ext>(itot*jtot*ktot);
    Optics_scat* ssa_asy_gpu = allocate_gpu<Optics_scat>(itot*jtot*ktot);

    copy_to_gpu(k_ext_gpu, k_ext.data(), itot*jtot*ktot);
    copy_to_gpu(ssa_asy_gpu, ssa_asy.data(), itot*jtot*ktot);

    // Output arrays. Copy them in order to enable restarts later.
    Float* surface_down_direct_count_gpu = allocate_gpu<Float>(itot*jtot);
    Float* surface_down_diffuse_count_gpu = allocate_gpu<Float>(itot*jtot);
    Float* surface_up_count_gpu = allocate_gpu<Float>(itot*jtot);
    Float* toa_down_count_gpu = allocate_gpu<Float>(itot*jtot);
    Float* toa_up_count_gpu = allocate_gpu<Float>(itot*jtot);
    Float* atmos_direct_count_gpu = allocate_gpu<Float>(itot*jtot*ktot);
    Float* atmos_diffuse_count_gpu = allocate_gpu<Float>(itot*jtot*ktot);

    copy_to_gpu(surface_down_direct_count_gpu, surface_down_direct_count.data(), itot*jtot);
    copy_to_gpu(surface_down_diffuse_count_gpu, surface_down_diffuse_count.data(), itot*jtot);
    copy_to_gpu(surface_up_count_gpu, surface_up_count.data(), itot*jtot);
    copy_to_gpu(toa_down_count_gpu, toa_down_count.data(), itot*jtot);
    copy_to_gpu(toa_up_count_gpu, toa_up_count.data(), itot*jtot);
    copy_to_gpu(atmos_direct_count_gpu, atmos_direct_count.data(), itot*jtot*ktot);
    copy_to_gpu(atmos_diffuse_count_gpu, atmos_diffuse_count.data(), itot*jtot*ktot);

    // Cloud dimensions
    std::vector<Int> cloud_mask_v(ktot);
    std::vector<Float> cloud_dims(2);

    Int* cloud_mask_v_gpu = allocate_gpu<Int>(ktot);
    Float* cloud_dims_gpu = allocate_gpu<Float>(2);

    copy_to_gpu(cloud_mask_v_gpu, cloud_mask_v.data(), ktot);
    copy_to_gpu(cloud_dims_gpu, cloud_dims.data(), 2);

    const int block_size_m = ktot;
    dim3 grid_m{1}, block_m{block_size_m};
    auto start_m = std::chrono::high_resolution_clock::now();

    cloud_mask_kernel<<<grid_m, block_m>>>(
            ssa_asy_gpu, cloud_mask_v_gpu,
            cloud_dims_gpu, dz_grid,
            itot, jtot, ktot);

    auto end_m = std::chrono::high_resolution_clock::now();
    double duration_m = std::chrono::duration_cast<std::chrono::duration<double>>(end_m - start_m).count();
    std::cout << "Duration cloud-mask computation: " << std::setprecision(5) << duration_m << " (s)" << std::endl;

    copy_from_gpu(cloud_dims.data(), cloud_dims_gpu, 2);
    std::cout<<cloud_dims[0]<<" - "<<cloud_dims[1]<<std::endl;

    //// RUN THE RAY TRACER ////
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
            photons_per_thread,
            toa_down_count_gpu, toa_up_count_gpu,
            surface_down_direct_count_gpu, surface_down_diffuse_count_gpu, surface_up_count_gpu,
            atmos_direct_count_gpu, atmos_diffuse_count_gpu,
            k_ext_gpu, ssa_asy_gpu,
            k_ext_null, k_ext_gas, surface_albedo,
            x_size, y_size, z_size,
            dx_grid, dy_grid, dz_grid,
            dir_x, dir_y, dir_z,
            itot, jtot, ktot,
            qrng_vectors_gpu, qrng_constants_gpu,
            cloud_dims_gpu);

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
            binary_file.write(reinterpret_cast<const char*>(ptr), size*sizeof(Float));
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
