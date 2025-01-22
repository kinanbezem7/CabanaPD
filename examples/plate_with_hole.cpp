/****************************************************************************
 * Copyright (c) 2022-2023 by Oak Ridge National Laboratory                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of CabanaPD. CabanaPD is distributed under a           *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "mpi.h"
#include <CabanaPD.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <fstream>
#include <iostream>

void plateWithHoleExample( const std::string filename )
{
    // Use default Kokkos spaces
    using exec_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename exec_space::memory_space;

    // Read inputs
    CabanaPD::Inputs inputs( filename );

    // Material parameters
    double rho0 = inputs["density"];
    double E = inputs["elastic_modulus"];
    double nu = 0.25; // Use bond-based model
    double K = E / ( 3 * ( 1 - 2 * nu ) );
    double G0 = inputs["fracture_energy"];
    double delta = inputs["horizon"];
    delta += 1e-10;

    // Discretization
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];
    std::array<int, 3> num_cells = inputs["num_cells"];
    int m = std::floor( delta /
                        ( ( high_corner[0] - low_corner[0] ) / num_cells[0] ) );
    int halo_width = m + 1;

    // Force model
    using model_type = CabanaPD::ForceModel<CabanaPD::PMB, CabanaPD::Fracture>;
    model_type force_model( delta, K, G0 );

    // Custom particle creation functor to create hole
    double hole_radius = inputs["hole_radius"];
    double center_x = ( high_corner[0] + low_corner[0] ) / 2.0;
    double center_y = ( high_corner[1] + low_corner[1] ) / 2.0;

    auto create_plate_with_hole =
        KOKKOS_LAMBDA( const int, const double pos[3] )
    {
        double dx = pos[0] - center_x;
        double dy = pos[1] - center_y;
        double distance = std::sqrt( dx * dx + dy * dy );
        return distance > hole_radius;
    };

    // Particle generation
    auto particles = std::make_shared<
        CabanaPD::Particles<memory_space, typename model_type::base_model>>(
        exec_space(), low_corner, high_corner, num_cells, halo_width );

    // Create particles with hole
    particles->createParticles( exec_space(), create_plate_with_hole );

    // Boundary conditions for tension
    double sigma0 = inputs["traction"];
    double dy = particles->dx[1];
    double b0 = sigma0 / dy;

    CabanaPD::RegionBoundary plane1( low_corner[0], high_corner[0],
                                     low_corner[1] - dy, low_corner[1] + dy,
                                     low_corner[2], high_corner[2] );
    CabanaPD::RegionBoundary plane2( low_corner[0], high_corner[0],
                                     high_corner[1] - dy, high_corner[1] + dy,
                                     low_corner[2], high_corner[2] );
    std::vector<CabanaPD::RegionBoundary> planes = { plane1, plane2 };

    auto particles_f = particles->getForce();
    auto particles_x = particles->getReferencePosition();

    // Create symmetric force BC in y-direction
    auto bc_op = KOKKOS_LAMBDA( const int pid )
    {
        auto p_f = particles_f.getParticleView( pid );
        auto p_x = particles_x.getParticle( pid );
        auto ypos = Cabana::get( p_x, CabanaPD::Field::ReferencePosition(), 1 );
        auto sign = std::abs( ypos ) / ypos;
        Cabana::get( p_f, CabanaPD::Field::Force(), 1 ) += b0 * sign;
    };

    auto bc =
        createBoundaryCondition( bc_op, exec_space{}, *particles, planes );

    // Initialize particles
    auto rho = particles->sliceDensity();
    auto nofail = particles->sliceNoFail();
    auto x = particles->sliceReferencePosition();

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        rho( pid ) = rho0;
        // Set the no-fail zone near loading boundaries
        if ( x( pid, 1 ) <= plane1.low_y + delta + 1e-10 ||
             x( pid, 1 ) >= plane2.high_y - delta - 1e-10 )
            nofail( pid ) = 1;
    };
    particles->updateParticles( exec_space{}, init_functor );

    // Run simulation
    // Add before creating solver
    // Calculate the size of each cell in the x and y directions
    double cell_size_x = ( high_corner[0] - low_corner[0] ) / num_cells[0];
    double cell_size_y = ( high_corner[1] - low_corner[1] ) / num_cells[1];

    // Use the smaller of the two dimensions for a conservative estimate
    double particle_size = std::min( cell_size_x, cell_size_y );

    // Calculate the circumference of the hole
    double circumference = 2 * M_PI * hole_radius;

    // Calculate the number of prenotches needed
    int num_prenotch =
        static_cast<int>( std::ceil( circumference / particle_size ) );

    // Create prenotch positions around the hole
    Kokkos::Array<Kokkos::Array<double, 3>, num_prenotch> notch_positions;
    Kokkos::Array<Kokkos::Array<double, 3>, num_prenotch> v1_array;
    Kokkos::Array<Kokkos::Array<double, 3>, num_prenotch> v2_array;

    double notch_length =
        particle_size * 2.0; // or some small length for radial cracks
    double thickness = high_corner[2] - low_corner[2];

    for ( int i = 0; i < num_prenotch; ++i )
    {
        double angle = 2.0 * M_PI * i / num_prenotch;
        double x = center_x + hole_radius * std::cos( angle );
        double y = center_y + hole_radius * std::sin( angle );
        notch_positions[i] = { x, y, low_corner[2] };

        // v1 defines radial prenotch length and direction
        v1_array[i] = { std::cos( angle ) * notch_length,
                        std::sin( angle ) * notch_length, 0.0 };

        // v2 defines thickness direction
        v2_array[i] = { 0.0, 0.0, thickness };
    }

    // Create the prenotch object
    CabanaPD::Prenotch<num_prenotch> prenotch( v1_array, v2_array,
                                               notch_positions );

    // Then modify the solver creation
    auto cabana_pd = CabanaPD::createSolverFracture<memory_space>(
        inputs, particles, force_model, bc, prenotch );
    cabana_pd->init_force();
    cabana_pd->run();
}

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );
    plateWithHoleExample( argv[1] );
    Kokkos::finalize();
    MPI_Finalize();
}
