import ..RL: AbstractRLEnv, id, seed!, reset!, step!, render, close!, seed_action_space!, sample_action_space!, obs_space_type, obs_space_shape, is_discrete_action_space, action_space_n, action_space_shape, action_space_low, action_space_high, max_episode_steps

abstract type GymEnv <: AbstractRLEnv end

struct GymContinuousEnv <: GymEnv
    pyenv::PyObject
    obs_space_type::Type
end

struct GymDiscreteEnv <: GymEnv
    pyenv::PyObject
    ob_type::Type
end

function make_gym_env(name::String)
    pyenv = gym.make(name)
    ob_type = typeof(pyenv.observation_space.sample())
    if pyisinstance(pyenv.action_space, gym.spaces.Discrete)
        return GymDiscreteEnv(pyenv, ob_type)
    else
        return GymContinuousEnv(pyenv, ob_type)
    end
end


id(env::GymEnv) = env.pyenv.unwrapped.spec.id
max_episode_steps(env::GymEnv) = env.pyenv.spec.max_episode_steps
seed!(env::GymEnv, seed) = env.pyenv.seed(seed)
function reset!(env::GymEnv)
    obs = env.pyenv.reset()
    # This is to handle LazyFrame object returned by gym.FrameStack wrapper:
    if env.ob_type <: Array && !isa(obs, Array)
        obs = numpy.asarray(obs)
    end
    if env.ob_type <: Array{UInt8, 3}  # an image
        if size(obs)[1] < size(obs)[2]  # channel is first dim
            obs = permutedims(obs, (3, 2, 1))  # make channel last dim
        end
    end
    return convert(env.ob_type, obs)
end

function step!(env::GymDiscreteEnv, action)
    obs, r, d, info = env.pyenv.step(action - 1)
    # This is to handle LazyFrame object returned by gym.FrameStack wrapper:
    if env.ob_type <: Array && !isa(obs, Array)
        obs = numpy.asarray(obs)
    end
    if env.ob_type <: Array{UInt8, 3}  # an image
        if size(obs)[1] < size(obs)[2]  # channel is first dim
            obs = permutedims(obs, (3, 2, 1))  # make channel last dim
        end
    end
    return convert(env.ob_type, obs), r, d, info
end

function step!(env::GymContinuousEnv, action)
    obs, r, d, info = env.pyenv.step(action)
    # This is to handle LazyFrame object returned by gym.FrameStack wrapper:
    if env.ob_type <: Array && !isa(obs, Array)
        obs = numpy.asarray(obs)
    end
    if env.ob_type <: Array{UInt8, 3}  # an image
        if size(obs)[1] < size(obs)[2]  # channel is first dim
            obs = permutedims(obs, (3, 2, 1))  # make channel last dim
        end
    end
    return convert(env.ob_type, obs), r, d, info
end

render(env::GymEnv) = env.pyenv.render()
close!(env::GymEnv) = env.pyenv.close()
seed_action_space!(env::GymEnv, seed) = env.pyenv.action_space.seed(seed)
sample_action_space!(env::GymDiscreteEnv) = env.pyenv.action_space.sample() + 1
sample_action_space!(env::GymContinuousEnv) = env.pyenv.action_space.sample()
is_discrete_action_space(env::GymEnv) = pyisinstance(env.pyenv.action_space, gym.spaces.Discrete)
action_space_n(env::GymDiscreteEnv) = env.pyenv.action_space.n
function obs_space_shape(env::GymEnv)
    shape = env.pyenv.observation_space.shape
    if env.ob_type <: Array{UInt8, 3}  # an image
        if shape[1] < shape[2]  # channel is first dim
            return (shape[3], shape[2], shape[1])  # make channel last dim
        end
    else
        return shape
    end
end
obs_space_type(env::GymEnv) = env.ob_type
action_space_shape(env::GymContinuousEnv) = env.pyenv.action_space.shape
action_space_low(env::GymContinuousEnv) = env.pyenv.action_space.low
action_space_high(env::GymContinuousEnv) = env.pyenv.action_space.high
