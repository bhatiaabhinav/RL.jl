function gym_wrap(env::GymEnv, gym_wrapper_pycls, args...; kwargs...)
    pyenv_wrapped = gym_wrapper_pycls(env.pyenv, args...; kwargs...)
    ob_type = typeof(pyenv_wrapped.observation_space.sample())
    if pyisinstance(pyenv_wrapped.action_space, gym.spaces.Discrete)
        return GymDiscreteEnv(pyenv_wrapped, ob_type)
    else
        return GymContinuousEnv(pyenv_wrapped, ob_type)
    end
end

function correct_capped_cubic_video_schedule(episode_id; video_interval=1000)
    episode_number = episode_id + 1 # So that the 1st, 8th, 27th, ... 1000th episode gets recorded instead of 1st, 2nd, 9th, 28th, .... 10001th episode since they correspond to episode id 0,1,8,27,..,1000.
    if episode_number >= video_interval
        return episode_number % video_interval == 0
    else
        return gym.wrappers.monitor.capped_cubic_video_schedule(episode_number)  # correct argument passed
    end
end

function gym_monitor_wrap(env::GymEnv, dir; video_interval=1000, force=false)
    return gym_wrap(env, gym.wrappers.Monitor, dir, force=force, video_callable=(ep_id) -> correct_capped_cubic_video_schedule(ep_id, video_interval=video_interval))
end

function gym_atari_wrap(env::GymDiscreteEnv; noop_max=30, frame_skip=4, frame_stack=4, episodic_life=false)
    @info "Wrapping with AtariPreprocessing" noop_max frame_skip episodic_life
    env = gym_wrap(env, gym.wrappers.AtariPreprocessing, noop_max=noop_max, frame_skip=frame_skip, terminal_on_life_loss=episodic_life)
    env = gym_wrap(env, gym.wrappers.FrameStack, frame_stack)
    return env
end
