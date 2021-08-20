
abstract type AbstractRLWrapper <: AbstractRLEnv end # Some wrapper written in Julia
import Random

id(env::AbstractRLWrapper) = id(env.env)
Random.seed!(env::AbstractRLWrapper, seed) = Random.seed!(env.env, seed)
reset!(env::AbstractRLWrapper) = reset!(env.env)
step!(env::AbstractRLWrapper, action) = step!(env.env, action)
render(env::AbstractRLWrapper, mode::String) = render(env.env, mode)
close!(env::AbstractRLWrapper) = close!(env.env)
seed_action_space!(env::AbstractRLWrapper, seed) = seed_action_space!(env.env, seed)
sample_action_space!(env::AbstractRLWrapper) = sample_action_space!(env.env)
obs_space_type(env::AbstractRLWrapper) = obs_space_type(env.env)
obs_space_shape(env::AbstractRLWrapper) = obs_space_shape(env.env)
is_discrete_action_space(env::AbstractRLWrapper) = is_discrete_action_space(env.env)
action_space_n(env::AbstractRLWrapper) = action_space_n(env.env)
action_space_shape(env::AbstractRLWrapper) = action_space_shape(env.env)
action_space_low(env::AbstractRLWrapper) = action_space_low(env.env)
action_space_high(env::AbstractRLWrapper) = action_space_high(env.env)
max_episode_steps(env::AbstractRLWrapper) = max_episode_steps(env.env)
