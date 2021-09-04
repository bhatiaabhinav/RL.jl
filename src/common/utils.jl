using Flux
using StatsBase

normal_prob(x, μ, σ) = (1 / (√(2π) * σ)) * exp(-0.5 * ((x - μ) / σ)^2)
log_nomal_prob(x, μ, σ) = -0.5 * ((x - μ) / σ)^2 - log(√(2π) * σ)

function copy_net!(from_net, to_net)
    @debug "copying net"
    Flux.loadparams!(to_net, params(from_net))
end


function preprocess(obs::Array)
    if typeof(obs) <: Array{UInt8}
        convert(Array{Float32}, obs) ./ 255
    else
        return convert(Array{Float32}, obs)
    end
end

function l2norm(x)
    sqrt(sum(x.^2))
end

function clip_gradients!(grads, clip_by_norm)
    if clip_by_norm < Inf
        for grad in grads
            if isnothing(grad)
                continue
            end
            norm = l2norm(grad)
            if norm > clip_by_norm
                grad .*= clip_by_norm / norm
            end
        end
    end
end

function onehot(s::AbstractVector, n_dims::Int=maximum(s))
    x = zeros(Float32, n_dims, length(s))
    for (col, row) in enumerate(s)
        x[row, col] = 1
    end
    return x
end

function linear_schedule(steps::Integer, over_steps::Integer, start_val::Real, final_val::Real)
    return clamp(start_val + (final_val - start_val) * steps / over_steps, min(start_val, final_val), max(start_val, final_val))
end

function gradient_wrt_params(paramss, loss_fn, args...; kwargs...)
    loss = 0
    grads = gradient(paramss) do
        loss = loss_fn(args...;  kwargs...)
    end
    return grads, loss
end

function safe_softmax(x; α=1)
    T = eltype(x)
    if α > 0
        max_x = maximum(x)
        _x = (x .- max_x) ./ T(α)
        exps = exp.(_x)
        return exps ./ sum(exps)
    else
        p = zeros(T, length(x))
        a = argmax(x)
        p[a] = 1
        return p
    end
end

categorical_sample(rng, values, probability_weights) = sample(rng, values, ProbabilityWeights(probability_weights))

boltzman_sample(rng, x; α=1) = α > 0 ? categorical_sample(rng, 1:length(x), safe_softmax(x, α=α)) : argmax(x)
