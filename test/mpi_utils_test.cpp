/*
 * Copyright (c) 2023 University of Michigan
 *
 */

#include "green/utils/mpi_utils.h"

#include <green/utils/timing.h>

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

template <typename T>
void run_test_on_shared(green::utils::shared_object<T>& shared, size_t data_size) {
  size_t total_size = 0;
  size_t local_size = shared.local_size();
  MPI_Reduce(&local_size, &total_size, 1, green::utils::mpi_type<size_t>::type, MPI_SUM, 0, green::utils::context.node_comm);
  if (!green::utils::context.node_rank) {
    REQUIRE(total_size == data_size);
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

  SECTION("Intra-node split") {
    MPI_Comm global      = green::utils::mpi_context::context.global;
    int      global_rank = green::utils::mpi_context::context.global_rank;
    MPI_Comm shared;
    int      shared_rank;
    int      shared_size;
    green::utils::setup_intranode_communicator(global, global_rank, shared, shared_rank, shared_size);
    REQUIRE(shared_size == green::utils::mpi_context::context.node_size);
    REQUIRE(shared_rank == green::utils::mpi_context::context.node_rank);
  }

  SECTION("Inter-node split") {
    MPI_Comm global      = green::utils::mpi_context::context.global;
    int      global_rank = green::utils::mpi_context::context.global_rank;
    MPI_Comm shared;
    int      shared_rank;
    int      shared_size;
    MPI_Comm inter;
    int      inter_rank;
    int      inter_size;
    green::utils::setup_intranode_communicator(global, global_rank, shared, shared_rank, shared_size);
    green::utils::setup_internode_communicator(global, global_rank, shared_rank, inter, inter_rank, inter_size);
    if (!shared_rank) {
      REQUIRE(inter_rank == 0);
      REQUIRE(inter_size == 1);
    } else {
      REQUIRE(inter_rank == -1);
      REQUIRE(inter_size == -1);
    }
  }

  SECTION("Emulate Device split") {
    MPI_Comm global      = green::utils::mpi_context::context.global;
    int      global_rank = green::utils::mpi_context::context.global_rank;
    int      global_size = green::utils::mpi_context::context.global_size;
    MPI_Comm shared;
    int      shared_rank;
    int      shared_size;
    MPI_Comm devices;
    int      devices_rank;
    int      devices_size;
    int      devCount_per_node = 2;
    int      devCount_total    = 2;
    green::utils::setup_intranode_communicator(global, global_rank, shared, shared_rank, shared_size);
    green::utils::setup_devices_communicator(global, global_rank, shared_rank, devCount_per_node, devCount_total, devices,
                                             devices_rank, devices_size);
    if (shared_rank == 0 || shared_rank == 1) {
      REQUIRE(devices_rank == shared_rank);
      REQUIRE(devices_size == 2);
    } else {
      REQUIRE(devices_rank == -1);
      REQUIRE(devices_size == -1);
    }
    REQUIRE_THROWS_AS(green::utils::setup_devices_communicator(global, global_rank, shared_rank, shared_size + 2, global_size + 2,
                                                               devices, devices_rank, devices_size),
                      green::utils::mpi_communicator_error);
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
    size_t array_size = 1003;
    SECTION("RValue array") {
      green::utils::shared_object shared_r(ref_array<double>{array_size});
      run_test_on_shared(shared_r, array_size);
    }

    ref_array<double>           shared_data(array_size);
    green::utils::shared_object shared(shared_data);
    run_test_on_shared(shared, shared_data.size());
  }

  SECTION("AllReduce") {
    MPI_Comm            global        = green::utils::mpi_context::context.global;
    int                 global_size   = green::utils::mpi_context::context.global_size;
    size_t              _nso          = 20;
    MPI_Datatype        dt_matrix     = green::utils::create_matrix_datatype<double>(_nso * _nso);
    MPI_Op              matrix_sum_op = green::utils::create_matrix_operation<double>();
    std::vector<double> G(100 * _nso * _nso);
    std::fill(G.begin(), G.end(), 1.0);
    green::utils::allreduce(MPI_IN_PLACE, G.data(), G.size() / (_nso * _nso), dt_matrix, matrix_sum_op, global);
    REQUIRE(std::all_of(G.begin(), G.end(), [global_size](double g) { return std::abs(g - 1.0 * global_size) < 1e-12; }));
  }

  SECTION("AllReduce Std Complex") {
    MPI_Comm                          global        = green::utils::mpi_context::context.global;
    int                               global_size   = green::utils::mpi_context::context.global_size;
    size_t                            _nso          = 20;
    MPI_Datatype                      dt_matrix     = green::utils::create_matrix_datatype<std::complex<double>>(_nso * _nso);
    MPI_Op                            matrix_sum_op = green::utils::create_matrix_operation<std::complex<double>>();
    std::vector<std::complex<double>> G(100 * _nso * _nso);
    std::fill(G.begin(), G.end(), std::complex<double>(1.0, 2.0));
    green::utils::allreduce(MPI_IN_PLACE, G.data(), G.size() / (_nso * _nso), dt_matrix, matrix_sum_op, global);
    REQUIRE(std::all_of(G.begin(), G.end(), [global_size](const std::complex<double>& g) {
      return (std::abs(g.real() - 1.0 * global_size) < 1e-12) && (std::abs(g.imag() - 2.0 * global_size) < 1e-12);
    }));
  }

  SECTION("Test Event Printing") {
    green::utils::timing statistic;
    double               s = MPI_Wtime();
    statistic.start("START");
    if(!green::utils::mpi_context::context.global_rank) {
      statistic.start("INNER");
      statistic.end();
    }
    statistic.start("INNER2");
    statistic.end();
    statistic.end();
    if(green::utils::mpi_context::context.global_rank) REQUIRE(statistic.event("START").children.size() == 1);
    statistic.print(MPI_COMM_WORLD);
    if(green::utils::mpi_context::context.global_rank) REQUIRE(statistic.event("START").children.size() == 2);
  }
}