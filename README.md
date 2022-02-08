# PSO-GPU

A basic particle swarm optimizer is implemented in Python and 
makes use of the GPU. Times are included
as a performance measure for comparison.

## Inspiration

For my undergraduate thesis, I implemented various PSO algorithms and found that Python has a 
[global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock), effectively limiting the multiprocessing 
I was attempting to do. After a reinforcement learning course I learned about PyTorch, and the use of GPU to speed up 
computations. Therefore, I tried to see if this GPU enhancement can be brought over to the PSO.


## Dependencies

These are the packages and Python version I used for implementation:

Package  | Version
------------- | -------------
TQDM  | 4.62.3
PyTorch  | 1.10.0+cu113
Python | 3.9

## Setup

```
git clone [insert project url here]
cd PSO-GPU
```

## Experimentation

### Hardware Specifications

This is the hardware of my computer, which was used for experimentation:

Component  | Type
------------- | -------------
CPU  | Intel i7-4930K (3.4GHz)
GPU  | Nvidia GTX 980
RAM | 16GB (1600 MHz)

### Optimization Functions
* [Rastrigin Function](https://www.sfu.ca/~ssurjano/rastr.html) 
* [Ackley Function](https://www.sfu.ca/~ssurjano/ackley.html)

### Standard Parameters
* Inertia: 0.729
* Cognitive: 1.49445
* Social: 1.49445
* Iterations: 10000
* Initialization Range: [-5.12, 5.12]

No velocity clamping, nor infeasibility if outside of range is implemented.

### Results

Set | Swarm Size | Dimensions
------------- | ------------- | -------------
1  | 100 | 100
2  | 500 | 100
3 | 100 | 500
4 | 1000 | 100
5 | 100 | 1000
6 | 1000 | 1000

The table above expresses 6 sets of parameter configurations that were altered during experimentation. Each set was
executed on each [optimization function](#Optimization-Functions), and on both CUDA and CPU.


Set | CUDA Time | CPU Time
------------- | ------------- | -------------
1 | 12.25s (1056 iter/s) | 8.7s (1149 iter/s)
2 | 9.97s (1002 iter/s) | 19.73s (506 iter/s)
3 | 9.55s (1047 iter/s) | 20.01s (499 iter/s)
4 | 9.99s (1000 iter/s) | 28.96s (345 iter/s)
5 | 9.54s (1048 iter/s) | 29.1s (343 iter/s)
6 | 11.35s (881 it/s) | 391.62s (25 iter/s)

The table above shows results from optimizing the rastrigin function. Clearly CPU overtakes when a lower swarm size and
dimension is used. However, as soon as both are dramatically increased, it's GPU that executes quicker. Notably, GPU is
consistent in execution times, with the lag in lower parameter values likely due to overhead initialization of tensors
and transfer over to the GPU.

Set | CUDA Time | CPU Time
------------- | ------------- | -------------
1 | 13s (952 iter/s) | 8.97s (1115 iter/s)
2 | 10.65s (939 iter/s) | 20.36s (491 iter/s)
3 | 10.42s (959 iter/s) | 20.31s (492 iter/s)
4 | 10.5s (952 iter/s) | 32.96s (303 iter/s)
5 | 10.75s (930 iter/s) | 32.39s (308 iter/s)
6 | 10.87s (920 it/s) | 378.22s (26 iter/s)

The table above shows results from optimizing the ackley function. Results are very similar from what is observed in the
previous table. This was merely to see if a change in function would lead to drastic change in runtime. However, even
though the ackley function is intuitively more complex to compute than rastrigin, set 6 had a lower runtime. This could 
be from a multitude of factors.

The point of this work was to implement a PSO to utilize the GPU, and perform some simple comparisons against sole CPU
usage. This was complete, and it shows that GPU vastly outperforms CPU when dimensionality and swarm size are increased.
If a low swarm size and dimensionality is needed, then it's not work using GPU since there is overhead in data transfer.

## Future Work

I believe that future work could be the implementation of some multi-objective particle swarm optimizers that try to
optimize for a pareto-optimal front. One such algorithm that I've already implemented is the multi-guided particle swarm
optimizer, however there are many more that could be tested on a GPU.