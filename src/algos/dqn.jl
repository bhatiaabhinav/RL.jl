using ..RL
using ..Common
using ..Common: copy_net!, linear_schedule, preprocess, gradient_wrt_params, clip_gradients!, boltzman_sample, safe_softmax
using ..Models
using Random
using Flux
using CUDA
using Flux.Optimise
using Logging
import BSON


mutable struct DQNAlgo <: AbstractRLAlgo
    double_dqn::Bool
    sarsa_dqn::Bool
    min_explore_steps::Int
    epsilon::Float32
    epsilon_schedule_steps::Int
    policy_temperature::Float32  # for soft policy
    use_mse_loss::Bool
    mb_size::Int
    lr::Float32
    sgd_steps_per_transition::Int
    grad_clip::Float32
    nsteps::Int
    exp_buff_len::Int
    train_interval_steps::Int
    target_copy_interval_steps::Int
    no_gpu::Bool
    save_model_interval_steps::Int
    load_model_path::Union{String, Nothing}
    eval_mode::Bool

    rng::MersenneTwister
    state_shape
    n_actions::Int
    exp_buff::ExperienceBuffer
    device
    qmodel::QModel
    target_qmodel::QModel
    optimizer
    max_ep_steps::Int
    function DQNAlgo(;double_dqn=false, sarsa_dqn=false, min_explore_steps=10000, epsilon=0.1, epsilon_schedule_steps=10000, policy_temperature=0, dqn_mse=false, mb_size=32, lr=0.0001, sgd_steps_per_transition=1, grad_clip=Inf32, nsteps=1, exp_buff_len=1000000, train_interval_steps=1, target_copy_interval_steps=2000, no_gpu=false, save_model_interval_steps=100000, load_model_path=nothing, eval_mode=false, kwargs...)
        d = new(double_dqn, sarsa_dqn, min_explore_steps, epsilon, epsilon_schedule_steps, policy_temperature, dqn_mse, mb_size, lr, sgd_steps_per_transition, grad_clip, nsteps, exp_buff_len, train_interval_steps, target_copy_interval_steps, no_gpu, save_model_interval_steps, load_model_path, eval_mode)
    end
end

RL.id(d::DQNAlgo) = "Deep Q Network (DQN) with double_dqn=$(d.double_dqn), sarsa_dqn=$(d.sarsa_dqn), mse_loss=$(d.use_mse_loss), nsteps=$(d.nsteps)"
RL.description(d::DQNAlgo) = "Based on Mnih et al. 2015.
Assumes discrete action space and continuous state space.
Policy temperature τ specifies the temperature of the target policy so that π(s,a) ∝ e^(q(s,a)/τ). τ=0 leads to a greedy policy.
The behavior policy is ϵ-target policy.
For enhanced stability, the value of the next state is always calculated using summation Σₐ(π(s',a) * q(s',a)) where π is the target policy, instead of taking the q value of an action sampled from π. This method, known as expected-SARSA (Sutton & Barto, Reinforcement Learning, 2018, Section 6.6).
sarsa_dqn: if true, the target policy becomes same as the behavior policy i.e. includes epsilon apart from the policy_temperature.
double_dqn: double learning to overcome maximization bias (Sutton & Barto, Reinforcement Learning, 2018, Section 6.7;  Hasselt et al., 2015).
dqn_mse: MSE loss is used instead of Huber loss.
Adam optimizer is used for q-updates.
Done/terminal signal is ignored for bellman update for episodes truncated due to a timelimit i.e. due to max_episode_steps(env).
In eval_mode, the code associated with training does not run. Note that epsilon is *not* ignored in eval_mode. Set it to zero explicitly for evaluation, if that's what you wnat."


function RL.init!(d::DQNAlgo, r::RLRun)
    seed_action_space!(r.env, r.seed)
    d.rng = MersenneTwister(r.seed)
    d.state_shape = obs_space_shape(r.env)
    d.n_actions = action_space_n(r.env)
    d.exp_buff = ExperienceBuffer{eltype(obs_space_type(r.env)),Int}(d.exp_buff_len, d.mb_size, nsteps=d.nsteps, gamma=r.gamma, ob_space_shape=d.state_shape)
    d.device = d.no_gpu ? cpu : gpu
    if isnothing(d.load_model_path)
        d.qmodel = QModel(d.state_shape, d.n_actions) |> d.device
        d.target_qmodel = QModel(d.state_shape, d.n_actions) |> d.device
    else
        d.qmodel = BSON.load(d.load_model_path, @__MODULE__)[:qmodel] |> d.device
        d.target_qmodel = BSON.load(d.load_model_path, @__MODULE__)[:qmodel] |> d.device
        @info "Loaded model" d.load_model_path
        !r.no_console_logs && with_logger(ConsoleLogger()) do
            @info "Loaded model" d.load_model_path
        end
    end
    copy_net!(d.qmodel, d.target_qmodel)
    d.optimizer = ADAM(d.lr)
    d.max_ep_steps = max_episode_steps(r.env)
    r.run_state[:policy_updates] = 0
    r.run_state[:epsilon] = 1
    rm(joinpath(r.logdir, "models"), force=true, recursive=true)
    mkpath(joinpath(r.logdir, "models"))
    @info "Initialized DQN" d.state_shape d.n_actions d.device d.qmodel d.optimizer d.max_ep_steps
    !r.no_console_logs && with_logger(ConsoleLogger()) do
        @info "Initialized DQN" d.state_shape d.n_actions d.device d.qmodel d.optimizer d.max_ep_steps
    end
end


function RL.act!(d::DQNAlgo, r::RLRun)
    ep = linear_schedule(r.total_steps - d.min_explore_steps, d.epsilon_schedule_steps, 1, d.epsilon)
    r.run_state[:epsilon] = ep
    if r.total_steps < d.min_explore_steps || (ep > 0 && rand(d.rng) < ep)
        return rand(d.rng, 1:d.n_actions)
    else
        obs = r.step_obs |> preprocess
        obs = reshape(obs, size(obs)..., 1) |> d.device
        q = d.qmodel(obs)[:, 1] |> cpu
        return boltzman_sample(d.rng, q, α=d.policy_temperature)
    end
end

function dqn_loss_fn(qmodel, mb_s, mb_q; dqn_mse=false)
    _loss_fn = dqn_mse ? Flux.Losses.mse : Flux.Losses.huber_loss
    return _loss_fn(qmodel(mb_s), mb_q)
end

function dqn_train(d::DQNAlgo, r::RLRun, sgd_steps)
    loss, mb_v, mb_ent = 0.0, 0.0, 0.0
    for step_no in 1:sgd_steps
        mb_s, mb_a, mb_r, mb_ns, mb_d, mb_h, _ =  random_experiences_unzipped(d.rng, d.exp_buff, d.mb_size)
        @debug "sampled minibatch" size(mb_s) size(mb_a) size(mb_r) size(mb_ns) size(mb_d) size(mb_h)
        mb_s = preprocess(mb_s) |> d.device
        mb_ns = preprocess(mb_ns) |> d.device
        mb_d = Array{Float32}(mb_d)

        ep = r.run_state[:epsilon]
        mb_q = d.qmodel(mb_s) |> cpu
        for i in 1:d.mb_size
            if d.sarsa_dqn
                probs = ep / d.n_actions .+ (1 - ep) .* safe_softmax(mb_q[:, i], α=d.policy_temperature)
            else
                probs = safe_softmax(mb_q[:, i], α=d.policy_temperature)
            end
            mb_v += sum(mb_q[:, i] .* probs) / d.mb_size
            mb_ent += any(probs .== 0) ? 0 : sum(.-probs .* log.(probs)) / d.mb_size
        end
        mb_target_nq = d.target_qmodel(mb_ns) |> cpu

        next_acting_q = mb_target_nq
        if d.double_dqn
            next_acting_q = d.qmodel(mb_ns) |> cpu
        end
        for i in 1:d.mb_size
            if d.sarsa_dqn
                probs = ep / d.n_actions .+ (1 - ep) .* safe_softmax(next_acting_q[:, i], α=d.policy_temperature)
            else
                probs = safe_softmax(next_acting_q[:, i], α=d.policy_temperature)
            end
            nv = sum(probs .* mb_target_nq[:, i])
            mb_q[mb_a[i], i] = mb_r[i] + (1 - mb_d[i]) * (r.gamma ^ mb_h[i]) * nv
        end

        mb_q = mb_q |> d.device
        grads, loss = gradient_wrt_params(params(d.qmodel), dqn_loss_fn, d.qmodel, mb_s, mb_q; dqn_mse=d.use_mse_loss)
        clip_gradients!(grads, d.grad_clip)
        update!(d.optimizer, params(d.qmodel), grads)
        r.run_state[:policy_updates] += 1
    end  
    r.run_state[:loss] = loss
    r.run_state[:mb_v] = mb_v
    r.run_state[:mb_entropy] = mb_ent
    return loss, mb_v, mb_ent
end

function RL.on_env_step!(d::DQNAlgo, r::RLRun)
    d.eval_mode && return

    push!(d.exp_buff, r.step_obs, r.step_action, r.step_reward, r.step_next_obs, (r.episode_steps < d.max_ep_steps) && r.step_terminal, r.step_info)

    if r.total_steps >= d.min_explore_steps && r.total_steps % d.train_interval_steps == 0
        dqn_train(d, r, d.sgd_steps_per_transition)
    end
    if r.total_steps % d.target_copy_interval_steps == 0
        copy_net!(d.qmodel, d.target_qmodel)
    end

    r.total_steps % d.save_model_interval_steps == 0  &&  save_model(d, r)
end

function RL.on_run_break!(d::DQNAlgo, r::RLRun)
    save_model(d, r)
end

function save_model(d::DQNAlgo, r::RLRun)
    qmodel = cpu(d.qmodel)
    BSON.@save joinpath(r.logdir, "models", "qmodel-$(r.total_steps)steps-$(r.total_episodes)episodes.bson") qmodel
    BSON.@save joinpath(r.logdir, "models", "qmodel-latest.bson") qmodel
    @info "Saved Model"
end
