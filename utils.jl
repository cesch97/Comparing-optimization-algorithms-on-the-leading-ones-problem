

### Statistics ###

function mean(x::Vector)
    sum(x) / length(x)
end

function std(x::Vector)
    x_mean = mean(x)
    sqrt(sum(broadcast(x -> (x - x_mean)^2, x)) / length(x))
end


### Evaluation ###

function eval_sol(solution::Array{Bool,1})
    #=
    The function compute the score of a solution
    on the leading-ones problem. The score is given 
    by the number of consecutive "trues" (ones) in a solution
    =#
    score = 0
    for i in solution
        if i
            score += 1
        elseif score != 0
            break
        end
    end
    score
end

function eval_algo(es::Function, params::Tuple, num_trials::Int)
    #=
    The function runs an optimization algorithm for a certain
    number of trials, each trial return the number of evaluations
    needed to find a solution. The mean result of the trials is returned
    =#
    mean([es(params...) for i in 1:num_trials])    
end


### Stochastic Gradient Descent ###

function update_params!(p::Array{Float64,1}, grad::Array{Float64,1}, lr::Float64)
    #=
    Run one step of SGD over the paramaters 'p'
    =#
    for i in eachindex(p)
        p[i] -= (grad[i] * lr)
    end
end

function update_params!(means::Array{Float64,2}, log_stds::Array{Float64,2},
                        mean_grad::Array{Float64,2}, log_std_grad::Array{Float64,2},
                        lr::Float64)
    #=
    Run one step of SGD over the paramaters (means, log_stds)
    =#
    for i in eachindex(means)
        means[i] -= (mean_grad[i] * lr)
        log_stds[i] -= (log_std_grad[i] * lr)
    end
end


# Plotting #

function plot_results(results::Array{Dict,1})
    #=
    Plot the results
    =#
    num_lines = length(results[1])
    points_per_line = length(results)
    p = plot(legend=:topleft)
    title!("Optimization algorithms comparison")
    xlabel!("solution length")
    ylabel!("number of evaluations")
    for (key, value) in results[1]
        x_axis = Int[]
        values = Float64[]
        for j in 1:points_per_line
            push!(x_axis, results[j][key].sol_len)
            push!(values, results[j][key].num_evals)
        end
        plot!(x_axis, values, label=key)
    end
    p
end