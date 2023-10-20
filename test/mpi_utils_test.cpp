/*
 * Copyright (c) 2023 University of Michigan
 *
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <chrono>
#include <thread>

#include "green/utils/mpi_utils.h"

TEST_CASE("MPI") {
  SECTION("Communicators split") {
    MPI_Comm global = MPI_COMM_WORLD;
    MPI_Comm shared = MPI_COMM_WORLD;
    MPI_Comm inter = MPI_COMM_WORLD;
    int rank, shared_rank, inter_rank;
    int size, shared_size, inter_size;
    MPI_Comm_rank(global, &rank);
    MPI_Comm_size(global, &size);
    green::utils::setup_communicators(global, rank, shared, shared_rank, shared_size, inter, inter_rank, inter_size);
  }

  SECTION("Shared memory routines") {
    MPI_Comm global = MPI_COMM_WORLD;
    MPI_Comm shared = MPI_COMM_NULL;
    MPI_Comm inter = MPI_COMM_NULL;
    int rank, shared_rank, inter_rank;
    int size, shared_size, inter_size;
    MPI_Comm_rank(global, &rank);
    MPI_Comm_size(global, &size);
    green::utils::setup_communicators(global, rank, shared, shared_rank, shared_size, inter, inter_rank, inter_size);
    double *data;
    MPI_Aint buffer_size = 1000;
    MPI_Win shared_win;
    green::utils::setup_mpi_shared_memory(&data, buffer_size, shared_win, shared, shared_rank);
    MPI_Win_fence(0, shared_win);
    if(!shared_rank) {
      std::fill(data, data + 999, 0.0);
      data[0] = 10.0;
    }
    MPI_Win_fence(0, shared_win);
    if(shared_rank)
      REQUIRE(std::abs(data[0] - 10) < 1e-12);
  }

  SECTION("Broadcast") {
    std::vector<double> x(100, 1.0);
    MPI_Comm global = MPI_COMM_WORLD;
    int rank;
    int size;
    MPI_Comm_rank(global, &rank);
    MPI_Comm_size(global, &size);
    if(!rank) std::fill(x.begin(), x.end(), 20);
    green::utils::broadcast(x.data(), x.size(), global, 0);
    if(rank) REQUIRE(std::all_of(x.begin(), x.end(), [] (double x) {return std::abs(x - 20)<1e-12;}));
    if(size>1) {
      if(rank==1) std::fill(x.begin(), x.end(), 30);
      green::utils::broadcast(x.data(), x.size(), global, 1);
      if(rank != 1) REQUIRE(std::all_of(x.begin(), x.end(), [] (double x) {return std::abs(x - 30)<1e-12;}));
    }

  }
}