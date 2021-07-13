using Logging

abstract type AbstractRLAlgo end


mutable struct RLRun
    name::String
    env::AbstractRLEnv
    algo::AbstractRLAlgo
    max_steps::Int
    max_episodes::Int
    seed::Int
    gamma::Float32
    logdir::String
    description::String

    start_time::Float64
    total_episodes::Int
    total_steps::Int
    total_reward::Float64
    interrupted::Bool
    total_time::Float64
    total_rpe::Float64

    episode_start_time::Float64
    episode_steps::Int
    episode_reward::Float32
    episode_discounted_return::Float32
    episode_info::Dict{Any,Any}
    episode_duration::Float64
    episode_steprate::Float64

    step_obs
    step_action
    step_next_obs
    step_reward::Float32
    step_terminal::Bool
    step_info::Dict{Any,Any}

    run_state::Dict{Symbol,Any} # Anything else not recorded as a field of this object. RL Algos can use it to store/exchange information.

    logger::AbstractLogger

    function RLRun(name::String, env::AbstractRLEnv, algo::AbstractRLAlgo; max_steps::Integer, max_episodes::Integer, seed::Integer=0, gamma::Real=0.99, logdir::String="logs/$(id(env))/$name", description::String="", kwargs...)
        rlrun = new(name, env, algo, max_steps, max_episodes, seed, gamma, logdir, description,  0, 0, 0, 0, false, 0, 0, 0, 0, 0, 0, Dict(), 0, 0, nothing, nothing, nothing, 0, true, Dict(), Dict{Symbol,Any}())
        mkpath(logdir)
        rlrun.logger = ConsoleLogger(open(joinpath(logdir, "rlrun_logs.txt"), "w+"))
        return rlrun
    end
end





id(algo::AbstractRLAlgo) = error("not implemented")
description(algo::AbstractRLAlgo) = ""
init!(algo::AbstractRLAlgo, rlrun::RLRun) = nothing
on_run_start!(algo::AbstractRLAlgo, rlrun::RLRun) = nothing
on_env_reset!(algo::AbstractRLAlgo, rlrun::RLRun) = nothing
act!(algo::AbstractRLAlgo, rlrun::RLRun) = nothing
on_env_step!(algo::AbstractRLAlgo, rlrun::RLRun) = nothing
on_env_terminal_step!(algo::AbstractRLAlgo, rlrun::RLRun) = nothing
on_run_break!(algo::AbstractRLAlgo, rlrun::RLRun) = nothing
on_env_close!(algo::AbstractRLAlgo, rlrun::RLRun) = nothing
on_run_finish!(algo::AbstractRLAlgo, rlrun::RLRun) = nothing




function interrupt!(rlrun::RLRun)
    rlrun.interrupted = true
end

id(rlrun::RLRun) = rlrun.name

description(rlrun::RLRun) = rlrun.description

function full_description(rlrun::RLRun)
    # TODO: add some key params to the full description
    contents = "RLRUN: $(id(rlrun)): $(description(rlrun))\n---------------------------\nENV: $(id(rlrun.env))\nALGO: $(description(rlrun.algo))\n"
    return contents
end

function run!(r::RLRun)
    with_logger(r.logger) do
        algo = r.algo
        env = r.env
        r.step_terminal = true
        moving_av_rpe = 0
        function should_stop()
            return r.interrupted || (r.total_steps >= r.max_steps) || (r.total_episodes >= r.max_episodes)
        end

        @info full_description(r)

        @info "Initializing algorithm"
        init!(algo, r)

        @info "Seeding environment" r.seed
        seed!(env, r.seed)

        @info "Starting run"
        r.start_time = time()
        on_run_start!(algo, r)
        while !should_stop()
            if r.step_terminal
                @debug "Resetting env"
                r.step_obs = copy(reset!(env))
                r.step_next_obs = nothing
                r.episode_steps = 0
                r.episode_reward = 0
                r.episode_discounted_return = 0
                r.episode_info = Dict()
                r.step_reward = 0
                r.step_terminal = false
                r.step_info = Dict()
                r.episode_start_time = time()
                on_env_reset!(algo, r)
            else
                r.step_obs = r.step_next_obs
            end
            @debug "Acting" r.step_obs
            r.step_action = act!(algo, r)
            if isnothing(r.step_action)
                error("algo returned no action")
            end
            r.step_next_obs, r.step_reward, r.step_terminal, r.step_info = step!(env, r.step_action)
            @debug "Stepped" r.step_action r.step_reward r.step_info
            r.step_next_obs = copy(r.step_next_obs)
            r.episode_steps += 1
            r.episode_reward += r.step_reward
            r.episode_discounted_return += (r.gamma^(r.episode_steps - 1)) * r.step_reward
            r.episode_duration = time() - r.episode_start_time
            r.episode_steprate = r.episode_steps / r.episode_duration
            r.total_steps += 1
            r.total_reward += r.step_reward
            r.total_time = time() - r.start_time
            on_env_step!(algo, r)
            if r.step_terminal
                @debug "terminal_signal"
                r.total_episodes += 1
                r.total_rpe = r.total_reward / r.total_episodes
                moving_av_rpe = 0.99 * moving_av_rpe + 0.01 * r.episode_reward
                on_env_terminal_step!(algo, r)
                @info "Episode finished" r.total_steps r.total_episodes r.episode_reward r.episode_steps r.total_rpe moving_av_rpe  r.run_state...
                with_logger(ConsoleLogger()) do
                    @info "Episode finished" r.total_steps r.total_episodes r.episode_reward r.episode_steps r.total_rpe moving_av_rpe r.episode_steprate r.run_state...
                end
            end
        end
        on_run_break!(algo, r)
        r.total_time = time() - r.start_time
        @info "Run stopped" r.total_time

        @info "Closing env"
        close!(env)
        on_env_close!(algo, r)

        @info "Run finished" r.total_steps r.total_reward r.total_episodes r.total_rpe
        with_logger(ConsoleLogger()) do
            @info "Run finished" r.total_time r.total_steps r.total_reward r.total_episodes r.total_rpe moving_av_rpe
        end
        on_run_finish!(algo, r)
        close(r.logger.stream)
    end
end


#TODO: Add documentation