

### Optimization Algorithms ###


# Evolution Strategies #

function one_plus_λ(sol_len::Int, pop_size::Int, mut_rate::Float64, max_evals::Int=10^5)
    #=
    Optimize a random solution by applying a mutation over 'n' copies of the solution
    and keeping only the one tht perform better for the next generation
    =#
    expert = rand(Bool, sol_len)
    max_fit = eval_sol(expert)
    num_evals = 1
    epoch = 0
    while true
        pop = [expert,]
        fitness = [max_fit,]
        for i in 1:pop_size
            mut = copy(expert)
            for j in 1:sol_len
                if rand() < mut_rate
                    mut[j] = rand(Bool)
                end
            end
            push!(pop, mut)
            push!(fitness, eval_sol(mut))
        end
        expert = pop[argmax(fitness)]
        max_fit = maximum(fitness)
        mean_fit = mean(fitness)
        num_evals += pop_size
        epoch += 1
        # @printf("%i -> Max: %i, mean: %.2f, num_evals: %i\n", epoch, max_fit, mean_fit, num_evals)
        if (max_fit == sol_len)
            return num_evals
        end
        if (num_evals >= max_evals)
            return NaN
        end
    end
    num_evals
end


function ga(sol_len::Int, pop_size::Int, mut_rate::Float64, tourn_size::Int, elite_rate::Float64, max_evals::Int=10^5)
    #=
    Initializing a population of 'n' random solutions and evolve the population through selection and mutation
    some of the best performers of a generation will skip directly to the next generation unchanged (elitism)
    =#
    pop = [rand(Bool, sol_len) for i in 1:pop_size]
    num_evals = 0
    epoch = 0
    elitism = Int(round(elite_rate * pop_size))
    local fitness
    while true
        if num_evals != 0
            fitness = vcat(fitness, eval_sol.(pop[elitism+1:end]))
            num_evals += (length(pop) - elitism)
        else
            fitness = eval_sol.(pop)
            num_evals += pop_size
        end
        ord_fit = reverse(sortperm(fitness))
        pop = pop[ord_fit]
        fitness = fitness[ord_fit]
        new_pop = pop[1:elitism]
        for i in elitism+1:pop_size
            sel_pool = [i,]
            fit_pool = [fitness[i],]
            pop_pool = filter(x -> x != i, [1:pop_size;])
            for j in 1:tourn_size-1
                sel = rand(pop_pool)
                filter!(x -> x != sel, pop_pool)
                push!(sel_pool, sel)
                push!(fit_pool, fitness[sel])
            end
            child = copy(pop[sel_pool[argmax(fit_pool)]])
            for k in 1:sol_len
                if rand() < mut_rate
                    child[k] = rand(Bool)
                end
            end
            push!(new_pop, child)
        end
        expert = pop[argmax(fitness)]
        max_fit = maximum(fitness)
        mean_fit = mean(fitness)
        fitness = fitness[1:elitism]
        epoch += 1
        # @printf("%i -> Max: %i, mean: %.2f, num_evals: %i\n", epoch, max_fit, mean_fit, num_evals)
        if (max_fit == sol_len)
            return num_evals
        end
        if (num_evals >= max_evals)
            return NaN
        end
        pop = new_pop
    end
end


# Policy Gradient #

function pg_bern_dist(sol_len::Int, pop_size::Int, lr::Float64, max_evals::Int=10^5)
    #=
    Creating a random generator made as an array of Bernoulli distributions and 
    updating the paramaters of the generator through stochastic gradient descent
    =#
    p = fill(0.5, sol_len)
    epoch = 0
    num_evals = 0
    while true
        p = broadcast(x -> isnan(x) ? x = 0.01 : x, p)
        p = max.(p, [0.01,])
        p = min.(p, [0.99])
        sol_gen = make_sol_gen(p)
        sample = [rand.(sol_gen) for i in 1:pop_size]
        pop = get_solution.(sample)
        fitness = eval_sol.(pop)
        num_evals += pop_size
        advantage = (fitness .- mean(fitness)) ./ std(fitness)
        grad = track_gradient(p, sample, advantage)
        update_params!(p, grad, lr)
        max_fit = maximum(fitness)
        mean_fit = mean(fitness)
        epoch += 1
        # @printf("%i -> Max: %i, mean: %.2f, num_evals: %i\n", epoch, max_fit, mean_fit, num_evals)
        if (max_fit == sol_len)
            return num_evals
        end
        if (num_evals >= max_evals)
            return NaN
        end
    end
end

function pg_norm_dist(sol_len::Int, pop_size::Int, lr::Float64, max_evals::Int=10^5)
    #=
    Creating a random generator made as a matrix of Normal distributions and 
    updating the paramaters of the generator (2 matricies means, log_stds) through stochastic gradient descent.
    To get a solution::Array{Bool} from a sample::Matrix{Float} the argmax function is applied to each row of the sample
    =#
    means = zeros(sol_len, 2)
    log_stds = zeros(sol_len, 2)
    epoch = 0
    num_evals = 0
    while true
        log_stds = broadcast(x -> isnan(x) ? x = -4.605 : x, log_stds)
        sol_gen = make_sol_gen(means, log_stds)
        sample = [rand.(sol_gen) for i in 1:pop_size]
        pop = get_solution.(sample)
        fitness = eval_sol.(pop)
        num_evals += pop_size
        advantage = (fitness .- mean(fitness)) ./ std(fitness)
        mean_grad, log_std_grad = track_gradient(means, log_stds, sample, advantage)
        update_params!(means, log_stds, mean_grad, log_std_grad, lr)
        max_fit = maximum(fitness)
        mean_fit = mean(fitness)
        epoch += 1
        # @printf("%i -> Max: %i, mean: %.2f, num_evals: %i\n", epoch, max_fit, mean_fit, num_evals)
        if (max_fit == sol_len)
            return num_evals
        end
        if (num_evals >= max_evals)
            return NaN
        end
    end
end
