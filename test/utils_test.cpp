/*
 * Copyright (c) 2023 University of Michigan.
 *
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <thread>

#include "green/utils/timing.h"

TEST_CASE("Timing Init") {
  SECTION("Test Start") {
    green::utils::execution_statistic_t statistic;
    statistic.add("START");
    statistic.start("START");
#ifndef NDEBUG
    REQUIRE_THROWS_AS(statistic.start("START"), green::utils::wrong_event_state);
#endif
    statistic.end();
    REQUIRE_NOTHROW(statistic.start("START"));
  }
}
TEST_CASE("Timing") {
  SECTION("Test Event Timing") {
    green::utils::execution_statistic_t statistic;
    statistic.add("START");
    double s = MPI_Wtime();
    statistic.start("START");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    statistic.end();
    double e = MPI_Wtime();
    REQUIRE(std::abs(statistic.event("START").duration - (e - s)) < 1e-3);
  }

  SECTION("Test Nesting Events") {
    green::utils::execution_statistic_t statistic;
    statistic.add("START");
    statistic.add("INNER");
    double s = MPI_Wtime();
    statistic.start("START");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    double si = MPI_Wtime();
    statistic.start("INNER");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    statistic.end();
    double ei = MPI_Wtime();
    statistic.end();
    double e = MPI_Wtime();
    REQUIRE(std::abs(statistic.event("START").duration - (e - s)) < 1e-3);
    REQUIRE(std::abs(statistic.event("INNER").duration - (ei - si)) < 1e-3);
    REQUIRE(statistic.event("INNER").parent == &(statistic.event("START")));
    REQUIRE(statistic.event("START").parent == nullptr);
    REQUIRE(statistic.event("START").children.size() == 1);
  }
}
