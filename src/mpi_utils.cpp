/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#include <green/utils/mpi_utils.h>

#include <stdexcept>

namespace green::utils {

  void setup_intranode_communicator(MPI_Comm global_comm, const int global_rank, MPI_Comm& intranode_comm, int& intranode_rank,
                                    int& intranode_size) {
    // Split _comm into several sub-communicators, one per shared memory domain,
    // In addition, one shared memory domain is equivalent to one node.
    if ((MPI_Comm_split_type(global_comm, MPI_COMM_TYPE_SHARED, global_rank, MPI_INFO_NULL, &intranode_comm)) != MPI_SUCCESS)
      throw mpi_communicator_error("Failed splitting shared-memory communicators.");
    MPI_Comm_rank(intranode_comm, &intranode_rank);
    MPI_Comm_size(intranode_comm, &intranode_size);
  }

  void setup_devices_communicator(MPI_Comm global_comm, int global_rank, int intranode_rank, int devCount_per_node,
                                  int devCount_total, MPI_Comm& devices_comm, int& devices_rank, int& devices_size) {
    if (intranode_rank < devCount_per_node) {
      int color = 0;
      MPI_Comm_split(global_comm, color, global_rank, &devices_comm);
      MPI_Comm_rank(devices_comm, &devices_rank);
      MPI_Comm_size(devices_comm, &devices_size);
      if (devices_size != devCount_total)
        throw mpi_communicator_error("Number of devices mismatches size of devices' communicator.");
    } else {
      MPI_Comm_split(global_comm, MPI_UNDEFINED, global_rank, &devices_comm);
      devices_rank = -1;
      devices_size = -1;
    }
  }

  void setup_internode_communicator(MPI_Comm global_comm, int global_rank, int intranode_rank, MPI_Comm& internode_comm, int& internode_rank, int& internode_size) {
    if (!intranode_rank) {
      MPI_Comm_split(global_comm, intranode_rank, global_rank, &internode_comm);
      MPI_Comm_rank(internode_comm, &internode_rank);
      MPI_Comm_size(internode_comm, &internode_size);
      if (!global_rank && internode_rank != global_rank) throw mpi_communicator_error("Root rank mismatched!");
    } else {
      MPI_Comm_split(global_comm, MPI_UNDEFINED, global_rank, &internode_comm);
      internode_rank = -1;
      internode_size = -1;
    }
  }

  void setup_communicators(MPI_Comm global_comm, int global_rank, MPI_Comm& intranode_comm, int& intranode_rank,
                           int& intranode_size, MPI_Comm& internode_comm, int& internode_rank, int& internode_size) {
    if ((MPI_Comm_split_type(global_comm, MPI_COMM_TYPE_SHARED, global_rank, MPI_INFO_NULL, &intranode_comm)) != MPI_SUCCESS)
      throw mpi_communicator_error("Failed to split shared-memory communicators.");
    MPI_Comm_rank(intranode_comm, &intranode_rank);
    MPI_Comm_size(intranode_comm, &intranode_size);
    if (!intranode_rank) {
      MPI_Comm_split(global_comm, intranode_rank, global_rank, &internode_comm);
      MPI_Comm_rank(internode_comm, &internode_rank);
      MPI_Comm_size(internode_comm, &internode_size);
      if (!global_rank && internode_rank != global_rank) throw mpi_communicator_error("Root rank mismatched!");
    } else {
      MPI_Comm_split(global_comm, MPI_UNDEFINED, global_rank, &internode_comm);
      internode_rank = -1;
      internode_size = -1;
    }
    MPI_Bcast(&internode_size, 1, MPI_INT, 0, intranode_comm);
    MPI_Bcast(&internode_rank, 1, MPI_INT, 0, intranode_comm);
    if (!global_rank) {
      std::cout << "Inter-node communicator has " << internode_size << " cores. Intra-node communicator has " << intranode_size
                << " cores." << std::endl;
      std::cout << std::flush;
    }
  }

  template void matrix_sum(double*, double*, int*, MPI_Datatype*);
  template void matrix_sum(std::complex<double>*, std::complex<double>*, int*, MPI_Datatype*);

}  // namespace green::utils