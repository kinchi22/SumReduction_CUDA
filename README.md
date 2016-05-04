# SumReduction_CUDA
Compare the elapsed time of Sum-Reduction.

Requirements
------------
CUDA toolkit & NVIDIA GPU supporting CUDA

Usage
------------
Command line options:
* ./SumReduction {# elements} {Method of Sum Reduction}
* Methods of Sum Reduction
  * CPU
    * cpu : run sum reduction by 'for loop' in CPU thread
    * cpu_ur : loop unrolling
  * GPU
    * g_atom : global atomicAdd
    * s_atom : shared atomicAdd
    * binary : binary tree reduction
    * shfl_a : shuffle + global atomicAdd
    * shfl_s : shuffle + shared atomicAdd
    * shfl_q : quantitative reduction with shuffle
  
You can use the script 'demo.sh' for test every methods.
