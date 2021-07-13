abstract type AbstractRLEnv end

id(env::AbstractRLEnv) = error("Not implemented")
seed!(env::AbstractRLEnv, seed) = @warn "not implemented"
reset!(env::AbstractRLEnv) = error("This function has not been implemented. Should return state")
step!(env::AbstractRLEnv, action) = error("This function has not been implemented. Should return next_state, reward, terminal_signal, info_dictionary")
render(env::AbstractRLEnv, mode::String) = nothing
close!(env::AbstractRLEnv) = nothing
seed_action_space!(env::AbstractRLEnv, seed) = @warn "not implemented"
sample_action_space!(env::AbstractRLEnv) = error("not implemented")
obs_space_type(env::AbstractRLEnv) = error("Not implemented")
obs_space_shape(env::AbstractRLEnv) = error("Not implemented")
is_discrete_action_space(env::AbstractRLEnv) = error("Not implemented")
action_space_n(env::AbstractRLEnv) = error("Not implemented")
action_space_shape(env::AbstractRLEnv) = error("Not implemented")
action_space_low(env::AbstractRLEnv) = error("Not implemented")
action_space_high(env::AbstractRLEnv) = error("Not implemented")
max_episode_steps(env::AbstractRLEnv) = error("Not implemented")