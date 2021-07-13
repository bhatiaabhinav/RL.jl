module Gym

using PyCall

const gym = PyNULL()
const numpy = PyNULL()
function __init__()
    copy!(gym, pyimport("gym"))
    copy!(numpy, pyimport("numpy"))
end

export make_gym_env, gym_wrap, gym_atari_wrap, gym_monitor_wrap

include("env.jl")
include("wrappers.jl")

end