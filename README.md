# Re-implementation for Learning Deep Kernels for Non-Parametric Two-Sample Tests ICML 2020
This is **partial** re-implementation for the [Learning Deep Kernels for Non-Parametric Two-Sample Tests, ICML 2020](https://arxiv.org/pdf/2002.09116)

## Dependencies
    bash run_depend.sh

## Run MNIST experiment
    python mnist_test.py

## Major modification compared to the original repo
We mainly re-organize the structure for the original codes, make it more readable and build a object-oriented structure.

We do not make any changes of the stable evaluation functions (i.e., hard-code statistical calculation).

## Test for MNIST dataset
* Results for Deep Kernel Method, metric is the average 10 runs for the Test Power.
  * n=100:
    * 0.457 (No available results in paper.)
  * n=200:
    * 0.997 (0.555 in paper)
  * n=500:
    * Unavailable due to computing resource limit.