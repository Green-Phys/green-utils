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
    Shared                       _object;
    size_t                       _size;
    typename Shared::value_type* _ref;
    size_t                       _local_size{};
    MPI_Win                      _win{};

    void                         assign_ptr() {
      _local_size = _size / mpi_context::context().node_size;
      _local_size += ((_size % mpi_context::context().node_size) > mpi_context::context().node_rank) ? 1 : 0;
      MPI_Aint l_size = _local_size;
      MPI_Aint g_size = _local_size;
      setup_mpi_shared_memory(&_ref, _local_size, g_size, _win, mpi_context::context());
      _object.set_ref(_ref);
    }

  public:
    template <typename... Args>
    explicit shared_object(size_t s1, Args... args) : _object(nullptr, s1, size_t(args)...), _size(_object.size()) {
      assign_ptr();
    }

    template <size_t N>
    explicit shared_object(const std::array<size_t, N>& shape) : _object(nullptr, shape), _size(_object.size()) {
      assign_ptr();
    }

    shared_object(Shared& obj) : _object(obj), _size(obj.size()) { assign_ptr(); }

    shared_object(Shared&& obj) : _object(obj), _size(obj.size()) { assign_ptr(); }

    shared_object(const shared_object& rhs) = delete;
    shared_object(shared_object&& rhs) :
        _object(rhs._object), _size(rhs._size), _ref(rhs._ref), _local_size(rhs._local_size), _win(rhs._win) {
      rhs._win = MPI_WIN_NULL;
      rhs._ref = nullptr;
    }

    shared_object& operator=(shared_object&& rhs) {
      _object     = rhs._object;
      _size       = rhs._size;
      _ref        = rhs._ref;
      _local_size = rhs._local_size;
      _win        = rhs._win;
      rhs._win    = MPI_WIN_NULL;
      rhs._ref    = nullptr;
      return *this;
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
