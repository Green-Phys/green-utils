/*
 * Copyright (c) 2023 University of Michigan
 *
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>

#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}