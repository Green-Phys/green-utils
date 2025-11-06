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
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include "except.h"

namespace green::utils {

  struct event_t {
    event_t() : start(0), duration(0), active(false), accumulate(true) {}
    event_t(double start_, double duration_) : start(start_), duration(duration_), active(false){};
    double                                                    start;
    double                                                    duration;
    bool                                                      active;
    bool                                                      accumulate; // accumulate time for subsequent measurements
    event_t*                                                  parent = nullptr;
    std::unordered_map<std::string, std::unique_ptr<event_t>> children;
  };

  inline void get_name(MPI_Comm comm, std::string& name) {
    int len  = name.length();
    MPI_Bcast(&len, 1, MPI_INT, 0, comm);
    if (len != name.length()) {
      name.resize(len);
    }
    MPI_Bcast(const_cast<char*>(name.data()), len, MPI_CHAR, 0, comm);
  }

  inline void print_event(const std::string& name, const std::string& prefix, const event_t& event) {
    std::stringstream ss;
    ss << std::setprecision(7) << std::fixed;
    ss << prefix << "Event '" << name << "' took ";
    std::cout << std::setw(45) << std::left << ss.str();
    ss.str("");
    ss << event.duration << " s." << std::endl;
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
    ss << std::setprecision(6) << std::fixed;
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
      ss << prefix << "Event '" << name << "' took";
      std::cout << std::setw(45) << std::left << ss.str();
      ss.str("");
      ss << std::fixed;
      ss << " " << std::setw(13) << max << " " << std::setw(13) << min << " " << std::setw(13) << avg / size << " s."
         << std::endl;
      std::cout << ss.str();
    }
    auto it = event.children.begin();
    for (int i = 0; i < event.children.size(); ++i) {
      std::string name = "";
      if(!rank) {
        name = it->first;
        std::advance(it, 1);
      }
      get_name(comm, name);
      const event_t& e = *event.children.at(name);
      print_event(comm, rank, size, name, prefix + "  ", e);
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
    timing(timing const&)         = delete;
    void operator=(timing const&) = delete;

    /**
     * @brief Register a root-level timing event by name without starting it.
     *
     * Creates the event entry in the internal root events map if it does not exist yet.
     * This call is idempotent and has no effect if the event is already present.
     *
     * Notes:
     * - This does not start timing. Use start(name) to begin measuring and end() to stop.
     * - Added events appear in print()/print(MPI_Comm) output even if never started (duration = 0).
     * - The event is registered at the root level; child events are created implicitly when
     *   start(name) is invoked while another event is active.
     *
     * @param name Unique identifier of the event to pre-register at the root level.
     */
    void add(const std::string& name) {
      if (_root_events.find(name) == _root_events.end()) {
        _root_events[name] = std::make_unique<event_t>(0.0, 0.0);
      }
    }

    /**
     * register the start point for the current frame for the event `name`
     * if called inside currently measured event, new event will be added as a child event into a current event.
     * `_current_event` will be set to a newly started event.
     *
     * @param name - event name to start time measurement
     * @param accumulate - whether to accumulate time for this event
     */
    void start(const std::string& name, bool accumulate=false) {
#ifndef NDEBUG
      if (_root_events.find(name) != _root_events.end() && _root_events[name]->active) {
        throw wrong_event_state("Event is already active");
      }
#endif
      if (_current_event) {
        // we have active event, start child event
        if (_current_event->children[name] == nullptr) _current_event->children[name] = std::make_unique<event_t>(0, 0);
        _current_event->children[name]->parent = _current_event;
        _current_event                         = _current_event->children[name].get();
        _current_event->accumulate             = accumulate;
      } else {
        // start root event
        if (_root_events[name] == nullptr) _root_events[name] = std::make_unique<event_t>(0, 0);
        _current_event = _root_events[name].get();
        _current_event->accumulate = accumulate;
      }
      _current_event->active = true;
      _current_event->start  = time();
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
      if (_current_event->accumulate) {
        _current_event->duration += time1 - _current_event->start;
      } else {
        _current_event->duration = time1 - _current_event->start;
      }
      _current_event->active = false;
      _current_event         = _current_event->parent;

    }


    /**
     * @brief reset the `duration` attribute of all child events of the current event
     * 
     */
    void reset() {
      if (!_current_event) return;
      for (auto& kv : _current_event->children) {
        kv.second->duration = 0.0;
        kv.second->active   = false;
      }
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
        std::cout << std::setw(43) << std::left << (_name + (_name.length() > 0 ? " " : "") + "timing: ") << std::setw(13)
                  << std::right << "max" << std::setw(13) << "min" << std::setw(13) << "avg" << std::endl;
      }
      auto old_precision = std::cout.precision();
      std::cout << std::setprecision(6);
      auto it = _root_events.begin();
      for (int i = 0; i < _root_events.size(); ++i) {
        std::string name = "";
        if(!id) {
          name = it->first;
          std::advance(it, 1);
        }
        get_name(comm, name);
        event_t& e = *_root_events[name];
        print_event(comm, id, np, name, "", e);
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
      if (_root_events.find(event_name) != _root_events.end()) {
        return *_root_events[event_name];
      }
      if (_current_event) {
        if (_current_event->children[event_name] == nullptr)
          _current_event->children[event_name] = std::make_unique<event_t>(0, 0);
        _current_event->children[event_name]->parent = _current_event;
        return *_current_event->children[event_name];
      }
      _root_events[event_name] = std::make_unique<event_t>(0.0, 0.0);
      return *_root_events[event_name];
    };

  private:
    // name of the timer
    std::string                    _name;
    // registered root timing events
    std::map<std::string, std::unique_ptr<event_t> > _root_events;
    event_t*                       _current_event = nullptr;

    /**
     * @return time in seconds since some arbitrary time in the past;
     */
    [[nodiscard]] static double    time() { return MPI_Wtime(); }

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
        if(!id) {
          name = it->first;
          std::advance(it, 1);
        }
        get_name(comm, name);
        if (events.find(name) == events.end()) {
          events[name] = std::make_unique<event_t>(0, 0);
        }
        event_t& e = *events[name];
        sync_events(comm, e.children);
      }
    }

  };
}  // namespace green::utils
#endif  // GREEN_UTILS_TIMING_H
