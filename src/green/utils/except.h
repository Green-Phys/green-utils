/*
 * Copyright (c) 2023 University of Michigan.
 *
 */

#ifndef GREEN_UTILS_EXCEPT_H
#define GREEN_UTILS_EXCEPT_H

#include <stdexcept>

namespace green::utils {
  class wrong_event_state : public std::runtime_error {
  public:
    explicit wrong_event_state(const std::string& what) : std::runtime_error(what) {}
  };

  class mpi_communicator_error : public std::runtime_error {
  public:
    explicit mpi_communicator_error(const std::string& what) : std::runtime_error(what) {}
  };
  class mpi_shared_memory_error : public std::runtime_error {
  public:
    explicit mpi_shared_memory_error(const std::string& what) : std::runtime_error(what) {}
  };
  class mpi_communication_error : public std::runtime_error {
  public:
    explicit mpi_communication_error(const std::string& what) : std::runtime_error(what) {}
  };
}  // namespace green::utils

#endif  // UTILS_EXCEPT_H
