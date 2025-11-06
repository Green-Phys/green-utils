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
    statistic.start("START");
#ifndef NDEBUG
    REQUIRE_THROWS_AS(statistic.start("START"), green::utils::wrong_event_state);
#endif
    statistic.end();
    REQUIRE_NOTHROW(statistic.start("START"));
  }
  SECTION("Test Event Timing") {
    green::utils::timing statistic;
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
    statistic.start("INNER");
    statistic.start("INNER");
    statistic.end();
    statistic.end();
    statistic.start("INNER2");
    statistic.end();
    statistic.end();
    REQUIRE_NOTHROW(statistic.print());
    REQUIRE_NOTHROW(statistic.print(MPI_COMM_WORLD));
  }

  SECTION("Test Nesting Events") {
    green::utils::timing statistic;
    double s = MPI_Wtime();
    statistic.start("START");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    double si = MPI_Wtime();
    statistic.start("INNER");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    double si2 = MPI_Wtime();
    statistic.start("INNER");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    statistic.end();
    statistic.end();
    double ei = MPI_Wtime();
    statistic.end();
    double e = MPI_Wtime();
    REQUIRE(std::abs(statistic.event("START").duration - (e - s)) < 1e-2);
    auto & inner_event = *statistic.event("START").children["INNER"];
    auto & inner2_event = *inner_event.children["INNER"];
    REQUIRE(std::abs(inner_event.duration - (ei - si)) < 1e-2);
    REQUIRE(std::abs(inner2_event.duration - (ei - si2)) < 1e-2);
    REQUIRE(inner_event.parent == &(statistic.event("START")));
    REQUIRE(statistic.event("START").parent == nullptr);
    REQUIRE(statistic.event("START").children.size() == 1);
    REQUIRE_NOTHROW(statistic.end());
    REQUIRE(statistic.event("UNKNOWN").parent == nullptr);
    statistic.start("TEST");
    REQUIRE(statistic.event("UNKNOWN2").parent == &statistic.event("TEST"));
    statistic.end();
  }

  SECTION("Test Accumulate") {
    // Test case when Accumulate is ON
    green::utils::timing statistic;
    double s1 = MPI_Wtime();
    statistic.start("ACCUMULATE ON", true);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    statistic.end();
    double e1 = MPI_Wtime();
    double duration1 = statistic.event("ACCUMULATE ON").duration;
    REQUIRE(std::abs(duration1 - (e1 - s1)) < 1e-2);

    double s2 = MPI_Wtime();
    statistic.start("ACCUMULATE ON", true);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    statistic.end();
    double e2 = MPI_Wtime();
    double duration2 = statistic.event("ACCUMULATE ON").duration;
    REQUIRE(std::abs(duration2 - ((e2 - s2) + (e1 - s1))) < 1e-2);

    // Test case when Accumulate is OFF
    statistic.start("ACCUMULATE OFF", false);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    statistic.end();
    double duration3 = statistic.event("ACCUMULATE OFF").duration;

    statistic.start("ACCUMULATE OFF", false);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    statistic.end();
    double duration4 = statistic.event("ACCUMULATE OFF").duration;
    REQUIRE(std::abs(duration4 - duration3) < 1e-2);
  }

  SECTION("Test Reset") {
    green::utils::timing statistic;
    statistic.start("ROOT");
    statistic.start("CHILD1");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    statistic.end();
    statistic.start("CHILD2");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    statistic.end();
    statistic.reset();
    statistic.end();

    double duration_root = statistic.event("ROOT").duration;
    double duration_child1 = statistic.event("ROOT").children["CHILD1"]->duration;
    double duration_child2 = statistic.event("ROOT").children["CHILD2"]->duration;

    REQUIRE(duration_child1 == 0.0);
    REQUIRE(duration_child2 == 0.0);
  }
}
