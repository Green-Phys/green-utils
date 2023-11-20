/*
 * Copyright (c) 2023 University of Michigan.
 *
 */

#ifndef GREEN_UTILS_MPI_UTILS_H
#define GREEN_UTILS_MPI_UTILS_H

#include <mpi.h>

#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>

#include "except.h"

namespace green::utils {

  template <typename prec>
  struct mpi_type {
    static MPI_Datatype type;
    static MPI_Datatype complex_type;
    static MPI_Datatype scalar_type;
  };

  template <typename prec>
  struct mpi_type<std::complex<prec>> {
    static MPI_Datatype type;
  };

  template <>
  inline MPI_Datatype mpi_type<double>::complex_type = MPI_C_DOUBLE_COMPLEX;
  template <>
  inline MPI_Datatype mpi_type<double>::scalar_type = MPI_DOUBLE;
  template <>
  inline MPI_Datatype mpi_type<double>::type = MPI_DOUBLE;
  template <>
  inline MPI_Datatype mpi_type<float>::complex_type = MPI_C_FLOAT_COMPLEX;
  template <>
  inline MPI_Datatype mpi_type<float>::scalar_type = MPI_FLOAT;
  template <>
  inline MPI_Datatype mpi_type<float>::type = MPI_FLOAT;
  template <>
  inline MPI_Datatype mpi_type<long>::type = MPI_LONG;
  template <>
  inline MPI_Datatype mpi_type<int>::type = MPI_INT;
  template <>
  inline MPI_Datatype mpi_type<size_t>::type = MPI_UNSIGNED_LONG;
  template <>
  inline MPI_Datatype mpi_type<std::complex<double>>::type = MPI_C_DOUBLE_COMPLEX;
  template <>
  inline MPI_Datatype mpi_type<std::complex<float>>::type = MPI_C_FLOAT_COMPLEX;

  void setup_intranode_communicator(MPI_Comm global_comm, int global_rank, MPI_Comm& intranode_comm, int& intranode_rank,
                                    int& intranode_size);

  void setup_devices_communicator(MPI_Comm global_comm, int global_rank, int intranode_rank, int devCount_per_node,
                                  int devCount_total, MPI_Comm& devices_comm, int& devices_rank, int& devices_size);

  void setup_internode_communicator(MPI_Comm global_comm, int global_rank, int intranode_rank, MPI_Comm& internode_comm, int& internode_rank, int& internode_size);

  void setup_communicators(MPI_Comm global_comm, int global_rank, MPI_Comm& intranode_comm, int& intranode_rank,
                           int& intranode_size, MPI_Comm& internode_comm, int& internode_rank, int& internode_size);

  /**
   * MPI runtime context
   */
  struct mpi_context {
    static inline mpi_context& context() {
      static mpi_context instance(MPI_COMM_WORLD);
      return instance;
    }

    mpi_context(MPI_Comm comm) : global(comm) {
      MPI_Comm_rank(global, &global_rank);
      MPI_Comm_size(global, &global_size);
      setup_communicators(global, global_rank, node_comm, node_rank, node_size, internode_comm, internode_rank, internode_size);
    }

    MPI_Comm global;
    int      global_rank;
    int      global_size;

    MPI_Comm node_comm;
    int      node_rank;
    int      node_size;

    MPI_Comm internode_comm;
    int      internode_rank;
    int      internode_size;
  };

  /**
   * Summation of memory contigious matrices.
   *
   * @tparam T - matrix element datatype
   * @param in - input matrix
   * @param inout - input-output matrix
   * @param len - number of matrices to sum up
   * @param dt - matrix datatype
   */
  template <typename T>
  void matrix_sum(T* in, T* inout, int* len, MPI_Datatype* dt) {
    int size;
    MPI_Type_size(*dt, &size);
    for (size_t i = 0; i < *len * static_cast<size_t>(size / sizeof(T)); ++i) {
      inout[i] += in[i];
    }
  }

  template <typename T>
  MPI_Datatype create_matrix_datatype(int N) {
    MPI_Datatype dt_matrix;
    MPI_Type_contiguous(N, mpi_type<T>::type, &dt_matrix);
    MPI_Type_commit(&dt_matrix);
    return dt_matrix;
  }

  template <typename T>
  MPI_Op create_matrix_operation() {
    MPI_Op matrix_sum_op;
    MPI_Op_create((MPI_User_function*)matrix_sum<T>, 5, &matrix_sum_op);
    return matrix_sum_op;
  }

  template <typename T>
  void allreduce(void* in, T* inout, int count, MPI_Datatype dt, MPI_Op op, MPI_Comm comm) {
    void* in_ptr = in;
    int   rank;
    MPI_Comm_rank(comm, &rank);
    if (in == MPI_IN_PLACE && rank != 0) in_ptr = inout;
    if (in == inout && rank == 0) in_ptr = MPI_IN_PLACE;

    int MPI_status = MPI_SUCCESS;
    if (!rank) {
      MPI_status = MPI_Reduce(in_ptr, inout, count, dt, op, 0, comm);
    } else {
      MPI_status = MPI_Reduce(in_ptr, in_ptr, count, dt, op, 0, comm);
    }
    if (MPI_status != MPI_SUCCESS) {
      std::cerr << "Rank " << rank << ": Reduction failed with error " << MPI_status << std::endl;
      throw mpi_communication_error("MPI_Allreduce failed.");
    }
    MPI_status = MPI_Bcast(inout, count, dt, 0, comm);
    if (MPI_status != MPI_SUCCESS) {
      std::cerr << "Rank " << rank << ": Broadcast failed with error " << MPI_status << std::endl;
      throw mpi_communication_error("MPI_Allreduce failed.");
    }
  }

  template <typename T>
  void setup_mpi_shared_memory(T** ptr_to_shared_mem, MPI_Aint& buffer_size, MPI_Win& shared_win, MPI_Comm intranode_comm,
                               int intranode_rank) {
    int disp_unit;
    // Allocate shared memory buffer (i.e. shared_win) on local process 0 of each shared-memory communicator (i.e. of each node)
    if (MPI_Win_allocate_shared((!intranode_rank) ? buffer_size * sizeof(T) : 0, sizeof(T), MPI_INFO_NULL, intranode_comm,
                                ptr_to_shared_mem, &shared_win) != MPI_SUCCESS)
      throw mpi_shared_memory_error("Failed allocating shared memory.");

    // This will be called by all processes to query the pointer to the shared area on local zero process.
    if (MPI_Win_shared_query(shared_win, 0, &buffer_size, &disp_unit, ptr_to_shared_mem) != MPI_SUCCESS)
      throw mpi_shared_memory_error("Failed extracting pointer to the shared area)");
    MPI_Barrier(intranode_comm);
  }

  template <typename T>
  void setup_mpi_shared_memory(T** ptr_to_shared_mem, MPI_Aint local_size, MPI_Aint& buffer_size, MPI_Win& shared_win,
                               MPI_Comm intranode_comm, int intranode_rank, int intranode_size) {
    int disp_unit;
    // Allocate shared memory buffer (i.e. shared_win) on local process 0 of each shared-memory communicator (i.e. of each node)
    if (MPI_Win_allocate_shared(local_size * sizeof(T), sizeof(T), MPI_INFO_NULL, intranode_comm, ptr_to_shared_mem,
                                &shared_win) != MPI_SUCCESS)
      throw mpi_shared_memory_error("Failed allocating shared memory.");

    // This will be called by all processes to query the pointer to the shared area on local zero process.
    if (MPI_Win_shared_query(shared_win, 0, &buffer_size, &disp_unit, ptr_to_shared_mem) != MPI_SUCCESS)
      throw mpi_shared_memory_error("Failed extracting pointer to the shared area)");
    MPI_Barrier(intranode_comm);
  }

  template <typename T>
  void setup_mpi_shared_memory(T** ptr_to_shared_mem, MPI_Aint local_size, MPI_Aint& buffer_size, MPI_Win& shared_win,
                               mpi_context& context) {
    int disp_unit;
    // Allocate shared memory buffer (i.e. shared_win) on local process 0 of each shared-memory communicator (i.e. of each node)
    if (MPI_Win_allocate_shared(local_size * sizeof(T), sizeof(T), MPI_INFO_NULL, context.node_comm, ptr_to_shared_mem,
                                &shared_win) != MPI_SUCCESS)
      throw mpi_shared_memory_error("Failed allocating shared memory.");

    // This will be called by all processes to query the pointer to the shared area on local zero process.
    if (MPI_Win_shared_query(shared_win, 0, &buffer_size, &disp_unit, ptr_to_shared_mem) != MPI_SUCCESS)
      throw mpi_shared_memory_error("Failed extracting pointer to the shared area)");
    MPI_Barrier(context.node_comm);
  }

  /**
   * Broadcast "object" from the root_rank to all processes in an internode communicator
   */
  template <typename T>
  void broadcast(T* object, size_t element_counts, MPI_Comm comm, int root_rank) {
    int size;
    MPI_Comm_size(comm, &size);
    if (size > 1) {
      // Split element_counts into chunks to avoid integer overflow inside MPI_Bcast
      size_t chunk_size = 1e8;
      for (size_t offset = 0; offset < element_counts; offset += chunk_size) {
        size_t mult = std::min(element_counts - offset, chunk_size);
        MPI_Bcast(object + offset, mult, mpi_type<T>::type, root_rank, comm);
      }
    }
  }

}  // namespace green::utils
#endif  // GREEN_UTILS_MPI_UTILS_H
