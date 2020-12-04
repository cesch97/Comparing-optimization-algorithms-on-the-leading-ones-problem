using Printf
using Hyperopt
using Distributions
using Plots
import Base.Threads.@threads
import Base.Threads.@spawn
include("utils.jl")
include("prob_utils.jl")
include("algos.jl")


#####################################################################
#                                                                   #
#   Comparing optimization algorithms on the leading-ones problem   #
#                                                                   #
#####################################################################


# Parameters #
sol_lengths = [20, 30, 50, 75]
opt_epochs  = 100
num_trials  = 5
max_evals   = 10^5
results_dir = "./results"
# - - - - - - - - - - - #


results = Array{Dict,1}(undef, length(sol_lengths))

@threads for i in eachindex(sol_lengths)

    sol_len = sol_lengths[i]

    # 1 + λ #
    one_plus_λ_th = @spawn begin
        ho = @hyperopt for i=opt_epochs,
                           pop_size=StepRange(4, 4, 128),
                           mut_rate=LinRange(0.01, 0.15, 15)
            eval_algo(one_plus_λ, (sol_len, pop_size, mut_rate, max_evals), num_trials)
        end
        opt_params, num_evals = minimum(ho)
        (sol_len=sol_len, params=opt_params, num_evals=num_evals, ho=ho)
    end

    # GA #
    ga_th = @spawn begin
        ho = @hyperopt for i=opt_epochs,
                           pop_size=StepRange(8, 4, 128),
                           mut_rate=LinRange(0.01, 0.15, 15),
                           tourn_size=StepRange(2, 1, 5),
                           elite_rate=LinRange(0.01, 0.5, 50)
            eval_algo(ga, (sol_len, pop_size, mut_rate, tourn_size, elite_rate, max_evals), num_trials)
        end
        opt_params, num_evals = minimum(ho)
        (sol_len=sol_len, params=opt_params, num_evals=num_evals, ho=ho)
    end

    # pg - bernoulli #
    pg_bern_th = @spawn begin
        ho = @hyperopt for i=opt_epochs,
                           pop_size=StepRange(4, 4, 128),
                           lr=LinRange(0.001, 0.1, 100)
            eval_algo(pg_bern_dist, (sol_len, pop_size, lr, max_evals), num_trials)
        end
        opt_params, num_evals = minimum(ho)
        (sol_len=sol_len, params=opt_params, num_evals=num_evals, ho=ho)
    end

    # pg - norm #
    pg_norm_th = @spawn begin
        ho = @hyperopt for i=opt_epochs,
                           pop_size=StepRange(4, 4, 128),
                           lr=LinRange(0.001, 0.1, 100)
            eval_algo(pg_norm_dist, (sol_len, pop_size, lr, max_evals), num_trials)
        end
        opt_params, num_evals = minimum(ho)
        (sol_len=sol_len, params=opt_params, num_evals=num_evals, ho=ho)
    end

    # saving results #
    results[i] = Dict("1 + λ" => fetch(one_plus_λ_th),
                      "GA" => fetch(ga_th),
                      "pg-bern" => fetch(pg_bern_th),
                      "pg-norm" => fetch(pg_norm_th))

    # logging #
    println("# sol_length: $sol_len")
    @printf(" > 1 + λ   -> %.2f\n", results[i]["1 + λ"].num_evals)
    @printf(" > GA      -> %.2f\n", results[i]["GA"].num_evals)
    @printf(" > pg-bern -> %.2f\n", results[i]["pg-bern"].num_evals)
    @printf(" > pg-norm -> %.2f\n", results[i]["pg-norm"].num_evals)
    println()
end


# plotting results #
plt = plot_results(results)
savefig(plt, results_dir*"/opt_algorithms_comparison.png")
