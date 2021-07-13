# RL.jl

A Reinforcement Learning Julia package that provides a framework and tools for developing new RL environments and RL algorithms. OpenAI gym environment are available via PyCall.

Popular RL algorithms included in this package:
- **DQN** (with n-step returns, double dqn and sarsa_dqn options). Please see src/Algos/dqn.jl for availble hyperparameters.


## Installation

1. Set up the python environment. To ensure that PyCall uses the desired python environment, activate the python environment, type `ENV["PYTHON"]="python"` in Julia REPL, and install PyCall using `Pkg.add("PyCall")`. In case PyCall was already installed, build it again using `Pkg.build("PyCall")`, since PyCall uses the python environment that was used to build it. After these steps, PyCall will continue to use this python environment, even if it is not activated.
2. Install `numpy`, `gym` and [`wandb`](https://wandb.ai/site) packages in the python environment.
3. Finally, to install RL.jl package, use the julia package manager to run `Pkg.add("https://github.com/bhatiaabhinav/RL.jl.git")`.

## Examples

The following script runs the Double-DQN algorithm with MSE loss and 3-step returns on CartPole-v0 gym envrionment.

```julia
using RL  # basic framework
using RL.Gym  # to use gym environments
using RL.Algos  # Popular algos like RandomPolicy, DQN, and convinient tools like Plotter and Wandb logger. See the files in src/algos directory for more details.
using Random
using PyCall

# Let's decide the configuration:
config = Dict(:run_name => "DDQN_Test_Run", :run_description => "checking out the new awesome RL package", :env_id => "CartPole-v0", :max_steps => 100000, :gamma => 0.99, :seed => 0, :no_gpu => true, :epsilon => 0.1, :dqn_mse => true, :double_dqn => true, :nsteps => 3, :logdir => "logs/CartPole-v0/DDQN_Test_Run")

# Make an OpenAI gym environment and wrap it in gym monitor
env = make_gym_env(config[:env_id])
env = gym_monitor_wrap(env, "$(config[:logdir])/gym_monitor", force=true)

# Make a composition of "algos" that participate in the RL Loop. An "algo" implements functions like `on_env_step`, `on_env_reset`, `act` etc. Only one "algo" can `act` in a composition of algos. See src/algo.jl for more details:
myalgo = CompositeRLAlgo(AbstractRLAlgo[  
    DQNAlgo(;config...),  # DQNAlgo accepts a lot of hyperparams. In this example, we are using the config dictionary to pass epsilon, dqn_mse, double_dqn, nsteps and no_gpu keyword arguments. no_gpu flag tells the algorithm to use CPU instead of CUDA for deep learning. Please see src/Algos/dqn.jl to see the default values of all hyperparams.
    PlotterAlgo()  # Continuously updates and displays a plot of rewards against episodes. This "algo" does not implement `act`. 
])


# comment out the following if you don't want wandb (https://wandb.ai/site) logging:
wandb = pyimport("wandb")
wandb.init(project=config[:env_id], name=config[:run_name], save_code=true, notes=config[:run_description], monitor_gym=true)
wandb.config.update(config)
push!(myalgo, WandbLoggerAlgo(wandb))  # Add WandbLoggerAlgo to myalgo composition. WandbLoggerAlgo logs certain metrics of RL run e.g., rewards, loss, etc. to wandb dashboard.


# run myalgo on env for `max_steps` steps:
Random.seed!(config[:seed])
rlrun = RLRun(config[:run_name], env, myalgo, max_steps=config[:max_steps], max_episodes=typemax(Int), seed=config[:seed], gamma=config[:gamma], logdir=config[:logdir], description=config[:run_description])  # setup the RL loop.
run!(rlrun)  # run the rl loop (see src/algo.jl). The rl-run logs will be written to the file config[:logdir]/rlrun_logs.txt
```

## Developing new RL Algos:
To develop a new RL algorithm, override the functions specified in src/algo.jl. See src/algos/dqn.jl for an example.

## Roadmap
- [ ] Add documentation about developing new RL envs and algos.
- [ ] Add more RL Algos.