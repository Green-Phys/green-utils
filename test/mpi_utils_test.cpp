/*
 * Copyright (c) 2023 University of Michigan
 *
 */

#include "green/utils/mpi_utils.h"

#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <thread>

#include "green/utils/mpi_shared.h"

template <typename T>
struct ref_array {
  using value_type = T;
  size_t _size;
  T*     _data;

  ref_array(size_t s) : _size(s) {}

  size_t   size() const { return _size; }
  T*       data() { return _data; }
  const T* data() const { return _data; }
  void     set_ref(T* ref) { _data = ref; }
};

TEST_CASE("MPI") {
  SECTION("Communicators split") {
    MPI_Comm global = MPI_COMM_WORLD;
    MPI_Comm shared = MPI_COMM_WORLD;
    MPI_Comm inter  = MPI_COMM_WORLD;
    int      rank, shared_rank, inter_rank;
    int      size, shared_size, inter_size;
    MPI_Comm_rank(global, &rank);
    MPI_Comm_size(global, &size);
    green::utils::setup_communicators(global, rank, shared, shared_rank, shared_size, inter, inter_rank, inter_size);
  }

  SECTION("Shared memory routines") {
    double*  data;
    MPI_Aint buffer_size = 1000;
    MPI_Win  shared_win;
    green::utils::setup_mpi_shared_memory(&data, buffer_size, shared_win, green::utils::context.node_comm,
                                          green::utils::context.node_rank);
    MPI_Win_fence(0, shared_win);
    if (!green::utils::context.node_rank) {
      std::fill(data, data + 999, 0.0);
      data[0] = 10.0;
    }
    MPI_Win_fence(0, shared_win);
    if (green::utils::context.node_rank) REQUIRE(std::abs(data[0] - 10) < 1e-12);
  }

  SECTION("Broadcast") {
    std::vector<double> x(100, 1.0);
    MPI_Comm            global = MPI_COMM_WORLD;
    int                 rank;
    int                 size;
    MPI_Comm_rank(global, &rank);
    MPI_Comm_size(global, &size);
    if (!rank) std::fill(x.begin(), x.end(), 20);
    green::utils::broadcast(x.data(), x.size(), global, 0);
    if (rank) REQUIRE(std::all_of(x.begin(), x.end(), [](double x) { return std::abs(x - 20) < 1e-12; }));
    if (size > 1) {
      if (rank == 1) std::fill(x.begin(), x.end(), 30);
      green::utils::broadcast(x.data(), x.size(), global, 1);
      if (rank != 1) REQUIRE(std::all_of(x.begin(), x.end(), [](double x) { return std::abs(x - 30) < 1e-12; }));
    }
  }

  SECTION("Shared wrapper") {
    ref_array<double>           shared_data(1003);
    green::utils::shared_object shared(shared_data);
    size_t                      total_size = 0;
    size_t                      local_size = shared.local_size();
    MPI_Reduce(&local_size, &total_size, 1, green::utils::mpi_type<size_t>::type, MPI_SUM, 0, green::utils::context.node_comm);
    if (!green::utils::context.node_rank) {
      REQUIRE(total_size == shared_data.size());
    }
    if (green::utils::context.node_size == 1) {
      return;
    }

    shared.fence();
    if (green::utils::context.node_rank == 1) {
      std::fill(shared.object().data(), shared.object().data() + shared.object().size(), 0.0);
    }
    shared.fence();
    REQUIRE(std::all_of(shared.object().data(), shared.object().data() + shared.object().size(),
                        [](double x) { return std::abs(x) < 1e-12; }));
    shared.fence();
    if (green::utils::context.node_rank == 1) {
      shared.object().data()[25] = 15;
    }
    shared.fence();
    if (green::utils::context.node_rank != 1) {
      REQUIRE(std::abs(shared.object().data()[25] - 15.0) < 1e-12);
    }
  }
}