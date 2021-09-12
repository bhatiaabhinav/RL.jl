using Random

mutable struct ExperienceBuffer{S,A}
    ob_space_shape
    states::Array{S}
    actions::Array{A}
    rewards::Array{Float32}
    next_states::Array{S}
    dones::Array{Bool}
    infos::Array{Dict}
    costs::Array{Float32}
    horizons::Array{Int32}
    capacity::Int
    nsteps::Int
    gamma::Float32
    count::Int
    idx::Int

    sample_size::Int
    sample_states::Array{S}
    sample_actions::Array{A}
    sample_rewards::Array{Float32}
    sample_next_states::Array{S}
    sample_dones::Array{Bool}
    sample_infos::Array{Dict}
    sample_costs::Array{Float32}
    sample_horizons::Array{Int32}


    function ExperienceBuffer{S,A}(capacity::Integer, sample_size::Integer; nsteps::Integer=1, gamma::Real=0.99, ob_space_shape) where {S,A}
        exp_buff = new{S,A}()
        exp_buff.ob_space_shape = ob_space_shape
        exp_buff.states = zeros(S, ob_space_shape..., capacity)
        exp_buff.actions = zeros(A, capacity)
        exp_buff.rewards = zeros(Float32, capacity)
        exp_buff.next_states =zeros(S, ob_space_shape..., capacity)
        exp_buff.dones = zeros(Bool, capacity)
        exp_buff.infos = Array{Dict}(undef, capacity)
        exp_buff.costs = zeros(Float32, capacity)
        exp_buff.horizons = zeros(Int32, capacity)
        exp_buff.capacity = capacity
        exp_buff.nsteps = nsteps
        exp_buff.gamma = gamma
        exp_buff.count = 0
        exp_buff.idx = 0

        exp_buff.sample_size = sample_size
        exp_buff.sample_states = zeros(S, ob_space_shape..., sample_size)
        exp_buff.sample_actions = zeros(A, sample_size)
        exp_buff.sample_rewards = zeros(Float32, sample_size)
        exp_buff.sample_next_states = zeros(S, ob_space_shape..., sample_size)
        exp_buff.sample_dones = zeros(Bool, sample_size)
        exp_buff.sample_infos = Array{Dict}(undef, sample_size)
        exp_buff.sample_costs = zeros(Float32, sample_size)
        exp_buff.sample_horizons = zeros(Int32, sample_size)
        return exp_buff
    end
end

function Base.empty!(exb::ExperienceBuffer)
    exb.count = 0
    exb.idx = 0
    return nothing
end

function get_prev_idx(exb::ExperienceBuffer, idx)
    if idx > 1
        return idx - 1
    else
        if exb.count == exb.capacity
            return exb.capacity
        else
            return 0
        end
    end
end

function propagate_back!(exb::ExperienceBuffer, idx::Integer, steps::Integer)
    r = exb.rewards[idx]
    ns = selectdim(exb.next_states, length(exb.ob_space_shape) + 1, idx)
    d = exb.dones[idx]
    for i in 1:steps
        idx = get_prev_idx(exb, idx)
        if idx <= 0 || exb.dones[idx]
            break
        end
        exb.rewards[idx] +=  exb.gamma^i * r
        selectdim(exb.next_states, length(exb.ob_space_shape) + 1, idx) .= ns
        exb.dones[idx] = d
        exb.horizons[idx] += 1
    end
    return nothing
end

function Base.push!(exb::ExperienceBuffer{S,A}, state::Array{S}, action::A, reward::Real, next_state::Array{S}, done::Bool, info::Dict; cost::Real=0, horizon::Integer=1) where {S,A}
    exb.idx = exb.idx % exb.capacity + 1
    selectdim(exb.states, ndims(exb.states), exb.idx) .= state
    exb.actions[exb.idx] = action
    exb.rewards[exb.idx] = reward
    exb.dones[exb.idx] = done
    selectdim(exb.next_states, ndims(exb.next_states), exb.idx) .= next_state
    exb.infos[exb.idx] = info
    exb.costs[exb.idx] = cost
    exb.horizons[exb.idx] = horizon
    exb.count = min(exb.count + 1, exb.capacity)
    propagate_back!(exb, exb.idx, exb.nsteps - 1)
    return nothing
end


function random_experiences_unzipped(rng::AbstractRNG, exb::ExperienceBuffer{S,A}, sample_size::Integer; get_costs=false) where {S,A}
    indices = rand(rng, 1:exb.count, sample_size)

    Threads.@threads for i in 1:sample_size
        selectdim(exb.sample_states, length(exb.ob_space_shape) + 1, i) .= selectdim(exb.states, length(exb.ob_space_shape) + 1, indices[i])
        selectdim(exb.sample_actions, 1, i) .= exb.actions[indices[i]]
        exb.sample_rewards[i] = exb.rewards[indices[i]]
        selectdim(exb.sample_next_states, length(exb.ob_space_shape) + 1, i) .= selectdim(exb.next_states, length(exb.ob_space_shape) + 1, indices[i])
        exb.sample_dones[i] = exb.dones[indices[i]]
        exb.sample_horizons[i] = exb.horizons[indices[i]]
        exb.sample_infos[i] = exb.infos[indices[i]]
        exb.sample_costs[i] = exb.costs[indices[i]]
    end

    if get_costs
        return exb.sample_states, exb.sample_actions, exb.sample_rewards, exb.sample_next_states, exb.sample_dones, exb.sample_horizons, exb.sample_infos, exb.sample_costs
    else
        return exb.sample_states, exb.sample_actions, exb.sample_rewards, exb.sample_next_states, exb.sample_dones, exb.sample_horizons, exb.sample_infos
    end
end

random_experiences_unzipped(exb::ExperienceBuffer, sample_size::Integer; get_costs=false) = random_experiences_unzipped(Random.GLOBAL_RNG, exb, sample_size; get_costs=get_costs)

function random_states(rng::AbstractRNG, exb::ExperienceBuffer{S,A}, sample_size::Integer) where {S,A}
    indices = rand(rng, 1:exb.count, sample_size)
    for i in 1:sample_size
        selectdim(exb.sample_states, length(exb.ob_space_shape) + 1, i) .= selectdim(exb.states, length(exb.ob_space_shape) + 1, indices[i])
    end
    return exb.sample_states
end

random_states(exb::ExperienceBuffer, sample_size::Integer) = random_states(Random.GLOBAL_RNG, exb, sample_size)


# TODO: Handle Discrete state envs
# TODO: Handle continuous action space
# TODO: Allow variable minibatch size
