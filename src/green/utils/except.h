/*
 * Copyright (c) 2023 University of Michigan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the “Software”), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify,
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
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
