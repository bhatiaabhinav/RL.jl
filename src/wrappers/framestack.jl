import ..RL: AbstractRLEnv, AbstractRLWrapper, obs_space_type, obs_space_shape, reset!, step!

mutable struct FrameStackWrapper{E <: AbstractRLEnv,S <: Array} <: AbstractRLWrapper
    env::E
    k::Int
    new_dim::Bool
    circular_q::Vector{S}
    latest_idx::Int
    cur_obs::Array
    function FrameStackWrapper(env::AbstractRLEnv; k=4, new_dim=false)
        E = typeof(env)
        S = obs_space_type(env)
        obs_example = zeros(eltype(S), obs_space_shape(env)...)
        return new{E,S}(env, k, new_dim, fill(obs_example, k), k, obs_example)
    end
end

function obs_space_shape(env::FrameStackWrapper)
    cur_shape = obs_space_shape(env.env)
    if env.new_dim
        new_shape = (cur_shape..., env.k)
    else
        new_shape = (cur_shape[1:end - 1]..., cur_shape[end] * env.k)
    end
    return new_shape
end

function stack_obs(env::FrameStackWrapper)
    cat_dim = length(size(env.circular_q[1]))
    if env.new_dim
        cat_dim += 1
    end
    return cat(env.circular_q[env.latest_idx + 1:end]..., env.circular_q[1:env.latest_idx]..., dims=cat_dim)
end

obs_space_type(env::FrameStackWrapper) = typeof(stack_obs(env))

function reset!(env::FrameStackWrapper)
    obs = reset!(env.env)
    fill!(env.circular_q, obs)
    env.latest_idx = env.k
    return stack_obs(env)
end

function step!(env::FrameStackWrapper, action)
    obs, r, d, info = step!(env.env, action)
    env.latest_idx = env.latest_idx % env.k + 1
    env.circular_q[env.latest_idx] .= obs
    return stack_obs(env), r, d, info
    return obs_new, r, d, info
end