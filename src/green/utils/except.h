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
}  // namespace green::utils

#endif  // UTILS_EXCEPT_H
