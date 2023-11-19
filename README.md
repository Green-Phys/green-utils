[![GitHub license](https://img.shields.io/github/license/Green-Phys/green-utils?cacheSeconds=3600&color=informational&label=License)](./LICENSE)
[![GitHub license](https://img.shields.io/badge/C%2B%2B-17-blue)](https://en.cppreference.com/w/cpp/compiler_support/17)

![symm](https://github.com/Green-Phys/green-utils/actions/workflows/test.yaml/badge.svg)
[![codecov](https://codecov.io/github/Green-Phys/green-utils/graph/badge.svg?token=W74X4XKDIZ)](https://codecov.io/github/Green-Phys/green-utils)

# green-utilities
Common utilities for Green projects

## MPI utilities

Often, when MPI application is running on modern multi-core cluster, in addition to global MPI communicator it requires to create two communiucators, namely,
communicator local for current node and communicator for purely internode communications. With `mpi_context` object, All this three communicators are easily accessible
as follows.

```cpp
// Acccess to global communicator
MPI_Comm global = mpi_context::context.global;

// Access to a size of a global communicator
int size = mpi_context::context.global_size;

// Acccess to a communicator local to a current node
MPI_Comm node = mpi_context::context.node_comm;

// Rank of the current cpu in the communicator local to a current node 
int node_rank = mpi_context::context.node_rank;
```

***
`green::utils::shared_object` is a wrapper around combination of data access object (such as ndarray that does not own memory) and MPI shared memory.
It allocates shared memory and stores MPI window for that memory region and stores user-defined object that orginezes access to that memory.


***

## Timing utilities

`green::utils::timing` provides a functionality for recording time spent in particular section of the code.
For that you need to surround it with `start(event name)` and `end()` calls. You can either use global instance of
a timer, or create a local copy. If another event starts within already started event block it will be considered as
inner event. Below is a simple example of recoding an event named `TEST` with one inner event `INNER`.

```cpp
using namespace green::utils;

// Obtain global timer
auto & timer = timing.get_instance();

// Start time recording for an event "TEST"

timer.start("TEST");

// do some calculation here

// start inner event "INNER"
timer.start("INNER");

// do more calculations

// end inner event
timer.end();

// do more calculation

// end topmose event

timer.end();

// print timing

timer.print();
```

The output for the code above will look like this:

```ShellSession
Execution statistics:
Event 'TEST' took 1.01648 s.
  Event 'INNER' took 0.51145 s.
```


# Acknowledgements

This work is supported by National Science Foundation under the award CSSI-2310582
