module RL

export RLRun, run!, CompositeRLAlgo

export AbstractRLEnv, AbstractRLWrapper, id, reset!, step!, render, close!, seed_action_space!, sample_action_space!, obs_space_shape, obs_space_type, is_discrete_action_space, action_space_n, action_space_shape, action_space_low, action_space_high, max_episode_steps
export AbstractRLAlgo, id, full_description, RLRun, interrupt!, run!
export CompositeRLAlgo

include("env.jl")
include("wrapper.jl")
include("algo.jl")
include("composite_algo.jl")

include("gym/Gym.jl")
include("wrappers/Wrappers.jl")

include("common/Common.jl")
include("models/Models.jl")
include("algos/Algos.jl")


end
