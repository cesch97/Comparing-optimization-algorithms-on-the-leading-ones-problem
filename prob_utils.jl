

### Solutions Generator ###

function make_sol_gen(p::Array{Float64,1})
    #=
    Given an array of probabilties (0 < p < 1)
    retuns an array of Bernoulli distributions
    =#
    sol_gen = Bernoulli[]
    for _p in p
        push!(sol_gen, Bernoulli(_p))
    end
    sol_gen
end

function make_sol_gen(means::Array{Float64,2}, log_stds::Array{Float64,2})
    #=
    Given a matrix of means and a matrix of log_stds
    return a matrix of Normal distributions
    =#
    sol_gen = Array{Normal,2}(undef, size(means))
    for i in 1:size(means,1)
        sol_gen[i, 1] = Normal(means[i, 1], exp(log_stds[i, 1]))
        sol_gen[i, 2] = Normal(means[i, 2], exp(log_stds[i, 2]))
    end
    sol_gen
end

function get_solution(sample::BitArray{1})
    #=
    Convert the sample from an array of Bernoulli distributions
    which is a BitArray{1} into a Bool array for the evaluation
    =#
    convert(Array{Bool,1}, sample)
end

function get_solution(sample::Array{Float64,2})
    #=
    Convert a sample from a matrix of Normal distributions which is a matrix of floats
    into a Bool array by applying the argamx function to each row of the sample
    =#
    sol = dropdims(mapslices(x::Vector -> argmax(x) - 1, sample, dims=2), dims=2)
    convert(Array{Bool,1}, sol)
end


### Probability Density Functions ###

function pdf_bern_dist(p::Float64, x::Bool)
    #=
    Return the probability of a Bernoulli distribution 
    parameterized by 'p' of sampling the value x
    =#
    (p^x) * ((1 - p)^(1-x))
end

function pdf_norm_dist(μ::Float64, log_σ::Float64, x::Float64)
    #=
    Return the probability of a Normal distribution 
    parameterized by (μ, log_σ) of sampling the value x
    =#
    σ = exp(log_σ)
    (1 / (σ * sqrt(2 * pi))) * exp((-0.5) * ((x - μ) / σ)^2)  
end


### Log Probability ###

function calc_log_prob(p::Array{Float64,1}, sample::BitArray{1})
    #=
    Return the log of the probability of sampling a value::Vector)
    from an Array of Bernoulli distributions parameterized by vector 'p'
    =#
    probs = pdf_bern_dist.(p, sample)
    prob = 1
    for i in eachindex(probs)
        prob *= probs[i]
    end
    log(prob)
end

function calc_log_prob(means::Array{Float64,2}, log_stds::Array{Float64,2}, sample::Array{Float64,2})
    #=
    Return the log of the probability of sampling a matrix of value 
    from a Matrix of Normal distributions parameterized by the matricies (means, log_stds)
    =#
    probs = pdf_norm_dist.(means, log_stds, sample)
    prob = 1
    for i in eachindex(probs)
        prob *= probs[i]
    end
    log(prob)
end


### Gradient Log Probability ###

function track_gradient(p::Array{Float64,1}, sample::Array{BitArray{1},1}, advantage::Array{Float64,1})
    #=
    Compute the value of the gradient of the function -mean(calc_log_prob.(p, sample) .* advantage)
    with respect of the parameter 'p'
    =#
    grad = Float64[]
    for i in eachindex(p)
        _g = Float64[]
        for j in eachindex(advantage)
            if sample[j][i]
                push!(_g, (1 / p[i]) * advantage[j])
            else
                push!(_g, -((1 / (1 - p[i])) * advantage[j]))
            end
        end
        push!(grad, mean(_g))
    end
    grad .* (-1)
end

function grad_mean_norm_dist(μ::Float64, log_σ::Float64, x::Float64)
    #=
    Compute the value of the gradient of the function calc_log_prob(μ, log_σ, x)
    with respect of the parameter 'μ'
    =#
    σ = exp(log_σ)
    (x - μ) / (σ^2)
end

function grad_log_std_norm_dist(μ::Float64, log_σ::Float64, x::Float64)
    #=
    Compute the value of the gradient of the function calc_log_prob(μ, log_σ, x)
    with respect of the parameter 'log_σ'
    =#
    σ = exp(log_σ)
    (-1) + ((x - μ) / σ)^2
end

function track_gradient(means::Array{Float64,2}, log_stds::Array{Float64,2}, sample::Array{Array{Float64,2},1}, advantage::Array{Float64,1})
    #=
    Compute the value of the gradient of the function -mean(calc_log_prob.(means, log_stds, samples) .* advantage)
    with respect of the parameters (means, log_stds)
    =#
    mean_grads = [grad_mean_norm_dist.(means, log_stds, sample[i]) for i in eachindex(advantage)]
    log_std_grads = [grad_log_std_norm_dist.(means, log_stds, sample[i]) for i in eachindex(advantage)]
    mean_grad = Array{Float64,2}(undef, size(means))
    log_std_grad = Array{Float64,2}(undef, size(means))
    for i in eachindex(means)
        _g_mean = Float64[]
        _g_log_std = Float64[]
        for j in eachindex(advantage)
            push!(_g_mean, mean_grads[j][i] * advantage[j])
            push!(_g_log_std, log_std_grads[j][i] * advantage[j])
        end
        mean_grad[i] = mean(_g_mean)
        log_std_grad[i] = mean(_g_log_std)
    end
    mean_grad * (-1), log_std_grad * (-1)
end