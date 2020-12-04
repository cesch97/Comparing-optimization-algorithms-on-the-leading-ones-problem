using Flux


# Policy Gradient #

function pg_bern_dist(sol_len::Int, pop_size::Int, lr::Float64, max_evals::Int=10^5)
    p = fill(0.5, sol_len)
    flux_ps = params(p)
    opt = Descent(lr)
    epoch = 0
    num_evals = 0
    while true
        p = broadcast(x -> isnan(x) ? x = 0.01 : x, p)
        p = max.(p, [0.01,])
        p = min.(p, [0.99])
        sol_gen = make_sol_gen(p)
        sample = [rand.(sol_gen) for i in 1:pop_size]
        pop = get_solution.(sample)
        fitness = evaluate.(pop)
        num_evals += pop_size
        advantage = (fitness .- mean(fitness)) ./ std(fitness)
        grad = gradient(flux_ps) do 
            -mean(calc_log_prob.([p,], sample) .* advantage)
        end
        Flux.update!(opt, flux_ps, grad)
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
    means = zeros(sol_len, 2)
    log_stds = zeros(sol_len, 2)
    flux_ps = params(means, log_stds)
    opt = Descent(lr)
    epoch = 0
    num_evals = 0
    while true
        log_stds = broadcast(x -> isnan(x) ? x = -4.605 : x, log_stds)
        sol_gen = make_sol_gen(means, log_stds)
        sample = [rand.(sol_gen) for i in 1:pop_size]
        pop = get_solution.(sample)
        fitness = evaluate.(pop)
        num_evals += pop_size
        advantage = (fitness .- mean(fitness)) ./ std(fitness)
        grad = gradient(flux_ps) do 
            -mean(calc_log_prob.([means,], [log_stds,], sample) .* advantage)
        end
        Flux.update!(opt, flux_ps, grad)
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