using ..RL

mutable struct WandbLoggerAlgo <: AbstractRLAlgo
    wandb
end

RL.id(w::WandbLoggerAlgo) = "WandbLogger"
RL.description(w::WandbLoggerAlgo) = "Logs selected fields of rlrun at the end of every episode using the given wandb object. Entries in rlrun.run_state are also logged."

function RL.init!(w::WandbLoggerAlgo, r::RLRun)
    w.wandb.config.update(Dict("algo_id" => id(r.algo)))
    w.wandb.save("$(r.logdir)/*")  # does not save subfolders
    w.wandb.save("$(r.logdir)/plots/*")  # due to plotter
    return nothing
end

function RL.on_env_terminal_step!(w::WandbLoggerAlgo, r::RLRun)
    w.wandb.log(Dict("Total/Episodes" => r.total_episodes, "Total/Steps" => r.total_steps, "Total/Reward" => r.total_reward, "Total/Time" => r.total_time, "Total/RPE" => r.total_rpe, "Total/RPE (Last $(r.moving_window_size) Episodes)" => r.moving_window_rpe,  "Episode/Steps" => r.episode_steps, "Episode/Reward" => r.episode_reward, "Episode/Discounted_Return" => r.episode_discounted_return, "Episode/Info" => r.episode_info, "Episode/SPS"=> r.episode_steprate, "Episode/FinalStep" => r.step_info), step=r.total_steps)
    w.wandb.log(r.run_state, step=r.total_steps)
    return nothing
end

function RL.on_run_finish!(w::WandbLoggerAlgo, r::RLRun)
    w.wandb.log(Dict("Total/Episodes" => r.total_episodes, "Total/Steps" => r.total_steps, "Total/Reward" => r.total_reward, "Total/Time" => r.total_time, "Total/RPE" => r.total_rpe, "Total/RPE (Last $(r.moving_window_size) Episodes)" => r.moving_window_rpe), step=r.total_steps)
    w.wandb.log(r.run_state, step=r.total_steps)
    w.wandb.save("$(r.logdir)/*")
    w.wandb.save("$(r.logdir)/plots/*")
    return nothing
end
