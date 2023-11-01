/*
 * Copyright (c) 2023 University of Michigan
 *
 */
#ifndef UTILS_MPI_SHARED_H
#define UTILS_MPI_SHARED_H

#include "mpi_utils.h"

namespace green::utils {
  template <typename Shared>
  class shared_object {
  private:
    Shared& _object;
    size_t  _size;
    size_t  _local_size;
    MPI_Win _win;

  public:
    shared_object(Shared& obj) : _object(obj), _size(obj.size()) {
      _local_size = _size / mpi_context::context().node_size;
      _local_size += ((_size % mpi_context::context().node_size) > mpi_context::context().node_rank) ? 1 : 0;
      MPI_Aint                     l_size = _local_size;
      MPI_Aint                     g_size = _local_size;
      typename Shared::value_type* ref;
      setup_mpi_shared_memory(&ref, _local_size, g_size, _win, mpi_context::context());
      _object.set_ref(ref);
    }

    virtual ~shared_object() {
      if (_win != MPI_WIN_NULL) MPI_Win_free(&_win);
    }

    void          fence(int assert = 0) { MPI_Win_fence(assert, _win); }

    size_t        local_size() const { return _local_size; }
    MPI_Win       win() const { return _win; }
    MPI_Win       win() { return _win; }
    size_t        size() const { return _size; }
    const Shared& object() const { return _object; }
    Shared&       object() { return _object; }
  };

#define context mpi_context::context()
}  // namespace green::utils

#endif  // UTILS_MPI_SHARED_H
