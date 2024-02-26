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
#ifndef GREEN_UTILS_TIMING_H
#define GREEN_UTILS_TIMING_H

#include <mpi.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>

#include "except.h"

namespace green::utils {

  struct event_t {
             event_t() : start(0), duration(0), active(false) {}
             event_t(double start_, double duration_) : start(start_), duration(duration_), active(false){};
    double   start;
    double   duration;
    bool     active;
    event_t* parent = nullptr;
    std::unordered_map<std::string, event_t*> children;
  };

  inline void print_event(const std::string& name, const std::string& prefix, const event_t& event) {
    std::stringstream ss;
    ss << std::setprecision(7) << std::fixed;
    ss << prefix << "Event '" << name << "' took " << event.duration << " s." << std::endl;
    std::cout << ss.str();
    for (auto& child : event.children) {
      print_event(child.first, prefix + "  ", *child.second);
    }
  }

  inline void print_event(MPI_Comm comm, int rank, int size, const std::string& name, const std::string& prefix,
                          const event_t& event) {
    double            max = event.duration;
    double            min = event.duration;
    double            avg = event.duration;
    std::stringstream ss;
    ss << std::setprecision(7) << std::fixed;
    if (!rank) {
      MPI_Reduce(MPI_IN_PLACE, &max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
      MPI_Reduce(MPI_IN_PLACE, &min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
      MPI_Reduce(MPI_IN_PLACE, &avg, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    } else {
      MPI_Reduce(&max, &max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
      MPI_Reduce(&min, &min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
      MPI_Reduce(&avg, &avg, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    }
    if (!rank) {
      ss << prefix << "Event '" << name << "' took max:" << max << " min: " << min << " avg: " << avg / size << " s."
         << std::endl;
      std::cout << ss.str();
    }
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
         timing(const std::string& name = "") : _name(name) {}

    // Delete possibility of coping timing object
         timing(timing const&)    = delete;
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
      std::cout << _name << (_name.length() > 0 ? " " : "") << "timing: " << std::endl;
      for (auto& kv : _root_events) {
        print_event(kv.first, "", *kv.second);
      }
      std::cout << "=====================" << std::endl;
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
      sync_events(comm, _root_events);
      if (!id) {
        std::cout << _name << (_name.length() > 0 ? " " : "") << "timing: " << std::endl;
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
    // name of the timer
    std::string                     _name;
    // registered timing events
    std::map<std::string, event_t>  _events;
    // registered root timing events
    std::map<std::string, event_t*> _root_events;
    event_t*                        _current_event = nullptr;

    /**
     * @return time in seconds since some arbitrary time in the past;
     */
    [[nodiscard]] static double     time() { return MPI_Wtime(); }

    /**
     * \brief Sync events between different cores, we assume that root core have all events
     *
     * \tparam EventMap map or unordered_map
     * \param comm MPI communicator to sync events over
     * \param events map of events
     */
    template <typename EventMap>
    void sync_events(MPI_Comm comm, EventMap& events) {
      int id, np;
      MPI_Comm_rank(comm, &id);
      MPI_Comm_size(comm, &np);
      int num_events = events.size();
      MPI_Bcast(&num_events, 1, MPI_INT, 0, comm);
      auto it = events.begin();
      for (int i = 0; i < num_events; ++i) {
        std::string name = "";
        int         len  = 0;
        if (!id) {
          name = it->first;
          len  = name.length();
          std::advance(it, 1);
        }
        MPI_Bcast(&len, 1, MPI_INT, 0, comm);
        if (id) {
          char* buf = new char[len];
          MPI_Bcast(buf, len, MPI_CHAR, 0, comm);
          name = std::string(buf, len);
          delete[] buf;
        } else {
          char* buf = new char[len];
          std::strcpy(buf, name.c_str());
          MPI_Bcast(buf, len, MPI_CHAR, 0, comm);
          delete[] buf;
        }
        if (events.find(name) == events.end()) {
          events[name] = &event(name);
        }
        event_t& e = event(name);
        sync_events(comm, e.children);
      }
    }
  };
}  // namespace green::utils
#endif  // GREEN_UTILS_TIMING_H
