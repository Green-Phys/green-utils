/*
 * Copyright (c) 2024 University of Michigan
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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <thread>

#include "green/utils/timing.h"
#include "green/utils/mpi_shared.h"

TEST_CASE("Timing") {
  SECTION("Test Start") {
    green::utils::timing statistic;
    statistic.add("START");
    statistic.start("START");
#ifndef NDEBUG
    REQUIRE_THROWS_AS(statistic.start("START"), green::utils::wrong_event_state);
#endif
    statistic.end();
    REQUIRE_NOTHROW(statistic.start("START"));
  }
  SECTION("Test Event Timing") {
    green::utils::timing statistic;
    statistic.add("START");
    double s = MPI_Wtime();
    statistic.start("START");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    statistic.end();
    double e = MPI_Wtime();
    REQUIRE(std::abs(statistic.event("START").duration - (e - s)) < 1e-3);
  }

  SECTION("Test Event Printing") {
    green::utils::timing statistic;
    double               s = MPI_Wtime();
    statistic.start("START");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    statistic.end();
    REQUIRE_NOTHROW(statistic.print());
    REQUIRE_NOTHROW(statistic.print(MPI_COMM_WORLD));
  }

  SECTION("Test Nesting Events") {
    green::utils::timing statistic;
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
    REQUIRE_NOTHROW(statistic.end());
    REQUIRE(statistic.event("UNKNOWN").parent == nullptr);
    statistic.start("TEST");
    REQUIRE(statistic.event("UNKNOWN2").parent == &statistic.event("TEST"));
    statistic.end();
  }
}
