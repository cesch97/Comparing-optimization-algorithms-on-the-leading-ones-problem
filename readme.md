# Comparing optimizations algorithms on the leading-ones problem

Trying different heuristic optimization algorithms with different set of hyperparameters to solve the leading-ones problem at different level of complexity.
  
### Prerequisites

To run this project you need: [Julia](https://julialang.org/) v1.5 or higher and the following Packages:
- Plots
- Printf
- Hyperopt
- Distributions
- Flux (optional)

## Getting Started

### The leading-ones problem 
In the leading ones problem we take an array of fixed length made of zeros and ones (or trues and falses) and the score is equal to the number of consecutive ones (trues) in the array. Our goal is to find the best solution to the problem (which is an array filled with ones) with less evaluation as possible (data efficiency). Even if it is an optimization problem becouse it's discrete we can't directly use differentiation to find an optimal solution so we need to look for gradient-free optimization algorithms.
  
### Implemented algorithms 
I've implemented four different derivative-free optimization algorithms, the first two are evolution strategies while the last two are based on deep reinforcement learning.
- 1 + λ
    - It's one of the simplest optimization algorithms out there, we start with a solution  that can be random or not then at each generation we create some copies of the solution and apply random noise to each of them. After evaluating each solution we keep only the best one for the next generation and the process repeats
- Genetic Algoritm
    - Instead of starting with a single solution we initialized an entire population of random solutions that is then evolved through selection and mutation
- Policy Gradient
    - In this class of algorithms we don't directly evolve the solution but instead we evolve a random generator to produce better and better solutions. We get a solution by sampling from the generator and then we increase or decrease the probability of the generator of sampling that value based on how the solution performed.
        - Bernoullli distribution
            - The random generator is represented as an array of Bernoulli distributions, to get a solution we sample directly from the generator
        - Normal distrbution
            - The random generator is represented as a matrix of Normal distributions, to get a solution we apply the argmax function to each row of the matrix sampled from the generator

### Algorithms evaluation
The algorithms are evaluated on solving the leading-ones problem with different solutions length (the size of the solution array). At each evaluation the hyper parameters of the algorithms are optimized using Hyperopt. Becouse a lot of random is involved to get more robust results we run each algorithm for a certain number of trials and then we keep the average of the results. Each algorithm runs till it finds the optimal solution or it reaches the maximum number of evaluations. The hyperparameters are optimized in order to get to the solution with less evaluation as possible.
  
### Implementation
For the gradient part I had two possibilities: 
- using Flux (auto-differentiation)
- differentiate manually and write some functions to compute the gradient

At first I used Flux but the implementation was quite slow so I opted to directly write a couple of functions to compute the gradient and that was much faster. I kept the Flux implementation of the policy-gradient algorithms in the "pg_algo_flux.jl" file. 
  
Thanks to Julia threading capabilities I could parallelize every hyperopt's job, practicaly I lunched all the jobs at the start and then I just let Julia do all the work and make good use of my cpu cores, speeding up the code by a lot (about 45 mins on my laptop).

### Results
In the **results** directory there are a plot which shows the minimum number of evaluations to get to the optimal solution for different length of the solution for each algorithm and the terminal output.

What can be seen in the results is that the data required by the evolution startegies scale up almost lineary with the complexity while for the policy gradient algorithms the data grow exponentialy. Also we can see that the simpler of the policy gradient algorithm the "pg-bern" which uses Bernoulli distributions performs better at low levels of complexity buf then performance falls drammaticaly (it's not able to reach a solution for the last solution length in a resenable amount of time) while the more complex one "pg-norm" that makes use of Normal distributions perform generaly worse then all the other algorithms but it can always reach a solution overtaking the "pg-bern" at 'sol_len = 75'.

### To contribute
If you have other algorithms to propose or a different problem to benchmark please send me a pull request. 

## Authors

* **Fabio Cescon** - [GitHub](https://github.com/cesch97)


## Acknowledgments

* A wonderful [project](https://github.com/d9w/evolution) made by Dennis G. Wilson. It's been my starting point into the world of evolutional computing.
* [Julia](https://julialang.org/)
* [Hyperopt](https://github.com/baggepinnen/Hyperopt.jl)

