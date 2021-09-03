using RL
using RL.Gym
using RL.Algos
using Random
using PyCall
using JSON: json

# decide config:
config = Dict(:run_name => "DQN_MSE Soft Loaded", :run_description => "DQN with MSE loss", :env_id => "CartPole-v0", :max_steps => 100000, :gamma => 0.99, :seed => 0, :no_gpu => true, :epsilon => 0.1, :dqn_mse => true, :nsteps => 1, :policy_temperature => 0.1, :sarsa_dqn=>false, :monitor_interval => 1000)
config[:logdir] = "logs/$(config[:env_id])/$(config[:run_name])"
rm(config[:logdir], force=true, recursive=true)
mkpath(config[:logdir])
open("$(config[:logdir])/config.json", "w") do f println(f, json(config, 4)) end

# make env:
env = make_gym_env(config[:env_id])
# env = gym_monitor_wrap(env, "$(config[:logdir])/gym_monitor", force=true, video_interval=config[:monitor_interval])

# make algo:
algo = CompositeRLAlgo(AbstractRLAlgo[
    DQNAlgo(;config..., save_model_interval_steps=50000),
    # PlotterAlgo(save_interval=10, no_display=true),
    # RenderAlgo()
])


# comment out if you don't want wandb logging:
# wandb = pyimport("wandb")
# wandb.init(project=config[:env_id], name=config[:run_name], save_code=true, notes=config[:run_description], monitor_gym=true)
# wandb.config.update(config)
# push!(algo, WandbLoggerAlgo(wandb))


# run algo on env
Random.seed!(config[:seed])
rlrun = RLRun(config[:run_name], env, algo, max_steps=config[:max_steps], max_episodes=typemax(Int), seed=config[:seed], gamma=config[:gamma], logdir=config[:logdir], description=config[:run_description], no_console_logs=false, logger_flush_interval=10)
run!(rlrun)
