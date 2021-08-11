using RL
using RL.Gym
using RL.Algos
using Random
using PyCall
using JSON: json

# decide config:
config = Dict(
    :run_name => "DQN_Mnih_3Step_MSE",
    :run_description => "DQN with 3-step returns and MSE loss. Mnih et al. 2015. neural architecure.",
    :env_id => "BreakoutNoFrameskip-v4",
    :max_steps => 10000000,
    :max_episodes => typemax(Int),
    :monitor_interval => 100,
    :seed => 0,
    :gamma => 0.99,
    :no_gpu => false,
    :min_explore_steps => 200000,
    :epsilon => 0.01,
    :epsilon_schedule_steps => 200000,
    :dqn_mse => true,
    :double_dqn => false,
    :sarsa_dqn => false,
    :nsteps => 3,
    :exp_buff_len => 200000,
    :grad_clip => Inf,
    :train_interval_steps => 4,
    :target_copy_interval_steps => 8000,
    :mb_size => 32,
    :lr => 0.0001
    )
config[:logdir] = "logs/$(config[:env_id])/$(config[:run_name])"

rm(config[:logdir], force=true, recursive=true)
mkpath(config[:logdir])
open("$(config[:logdir])/config.json", "w") do f println(f, json(config, 4)) end

# make env:
env = make_gym_env(config[:env_id])
env = gym_monitor_wrap(env, "$(config[:logdir])/gym_monitor", force=true, video_interval=config[:monitor_interval])
env = gym_atari_wrap(env)

# make algo:
algo = CompositeRLAlgo(AbstractRLAlgo[
    DQNAlgo(;config...),
    PlotterAlgo(save_interval=config[:monitor_interval])
])


# comment out if you don't want wandb logging:
wandb = pyimport("wandb")
wandb.init(project=config[:env_id], name=config[:run_name], save_code=true, notes=config[:run_description], monitor_gym=true)
wandb.config.update(config)
push!(algo, WandbLoggerAlgo(wandb))


# run algo on env
Random.seed!(config[:seed])
rlrun = RLRun(config[:run_name], env, algo, max_steps=config[:max_steps], max_episodes=config[:max_episodes], seed=config[:seed], gamma=config[:gamma], logdir=config[:logdir], description=config[:run_description])
run!(rlrun)
