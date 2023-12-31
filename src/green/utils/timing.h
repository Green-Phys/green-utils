/*
 * Copyright (c) 2023 University of Michigan.
 *
 */

#ifndef GREEN_UTILS_TIMING_H
#define GREEN_UTILS_TIMING_H

#include <mpi.h>

#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>

#include "except.h"

namespace green::utils {

  struct event_t {
    event_t() : start(0), duration(0), active(false) {}
    event_t(double start_, double duration_) : start(start_), duration(duration_), active(false){};
    double                                    start;
    double                                    duration;
    bool                                      active;
    event_t*                                  parent = nullptr;
    std::unordered_map<std::string, event_t*> children;
  };

  inline void print_event(const std::string& name, const std::string& prefix, const event_t& event) {
    std::cout << prefix << "Event '" << name << "' took " << event.duration << " s." << std::endl;
    for (auto& child : event.children) {
      print_event(child.first, prefix + "  ", *child.second);
    }
  }

  inline void print_event(MPI_Comm comm, int rank, int size, const std::string& name, const std::string& prefix,
                          const event_t& event) {
    double max = event.duration;
    double min = event.duration;
    double avg = event.duration;
    if (!rank) {
      MPI_Reduce(MPI_IN_PLACE, &max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
      MPI_Reduce(MPI_IN_PLACE, &min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
      MPI_Reduce(MPI_IN_PLACE, &avg, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    } else {
      MPI_Reduce(&max, &max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
      MPI_Reduce(&min, &min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
      MPI_Reduce(&avg, &avg, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    }
    if (!rank)
      std::cout << prefix << "Event '" << name << "' took max:" << max << " min: " << min << " avg: " << avg / size << " s."
                << std::endl;
    for (auto& child : event.children) {
      print_event(comm, rank, size, child.first, prefix + "  ", *child.second);
    }
  }

  /**
   * @brief timing class. measure time for user defined event in the code
   *
   */
  class timing {
  public:
    /**
     * Singleton pattern for time profiling object
     * @return
     */
    static timing& get_instance() {
      static timing instance;  // Guaranteed to be destroyed.
                               // Instantiated on first use.
      return instance;
    }

  public:
    /**
     * Default constructor is public for testing
     */
    timing()                      = default;

    // Delete possibility of coping timing object
    timing(timing const&)         = delete;
    void operator=(timing const&) = delete;

    void add(const std::string& name) {
      if (_events.find(name) == _events.end()) {
        _events[name] = event_t(0.0, 0.0);
      }
    }

    /**
     * register the start point for the current frame for the event `name`
     * if called inside currently measured event, new event will be added as a child event into a current event.
     * `_current_event` will be set to a newly started event.
     *
     * @param name - event name to start time measurement
     */
    void start(const std::string& name) {
#ifndef NDEBUG
      if (_events[name].active) {
        throw wrong_event_state("Event is already active");
      }
#endif
      if (_current_event) {
        _events[name].parent           = _current_event;
        _current_event->children[name] = (&_events[name]);
      } else {
        _root_events[name] = &_events[name];
      }
      _events[name].active = true;
      _current_event       = &_events[name];
      _events[name].start  = time();
    }

    /**
     * Finish time measurement for current event and update its time.
     * New value for `_current_event` will be set to its parent.
     * For all root event parent is nullptr.
     *
     */
    void end() {
      if (!_current_event) {
        return;
      }
      double time1 = time();
      _current_event->duration += time1 - _current_event->start;
      _current_event->active = false;
      _current_event         = _current_event->parent;
    }

    /**
     * Print statistics for all observed events
     */
    void print() {
      std::cout << "Runtime statistics:" << std::endl;
      auto old_precision = std::cout.precision();
      std::cout << std::setprecision(5);
      for (auto& kv : _root_events) {
        print_event(kv.first, "", *kv.second);
      }
      std::cout << "=====================" << std::endl;
      std::cout << std::setprecision(old_precision);
    }

    /**
     * Print statistics for all observed events. min, max and averaged time across all MPI processes within a given
     * MPI communicator will be computed and printed.
     *
     * @param comm - MPI communicator
     */
    void print(MPI_Comm comm) {
      int id, np;
      MPI_Comm_rank(comm, &id);
      MPI_Comm_size(comm, &np);
      if (!id) {
        std::cout << "Runtime statistics: " << std::endl;
      }
      auto old_precision = std::cout.precision();
      std::cout << std::setprecision(6);
      for (auto& kv : _root_events) {
        print_event(comm, id, np, kv.first, "", *kv.second);
      }
      if (!id) {
        std::cout << "=====================" << std::endl;
      }
      std::cout << std::setprecision(old_precision);
    }

    /**
     * Return timing event
     * @param event_name - event name
     * @return event by name
     */
    event_t& event(const std::string& event_name) {
      if (_events.find(event_name) != _events.end()) {
        return _events[event_name];
      }
      _events[event_name] = event_t(0.0, 0.0);
      if (_current_event) _events[event_name].parent = _current_event;
      return _events[event_name];
    };

  private:
    // registered timing events
    std::map<std::string, event_t>  _events;
    // registered root timing events
    std::map<std::string, event_t*> _root_events;
    event_t*                        _current_event = nullptr;

    /**
     * @return time in seconds since some arbitrary time in the past;
     */
    [[nodiscard]] static double     time() { return MPI_Wtime(); }
  };
}  // namespace green::utils
#endif  // GREEN_UTILS_TIMING_H
