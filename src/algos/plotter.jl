using ..RL
using Plots

mutable struct PlotterAlgo <: AbstractRLAlgo
    episode_rewards::Vector{Float32}
    save_interval::Int
    no_display::Bool
    PlotterAlgo(;save_interval=1000, no_display=false) = new(Float32[], save_interval, no_display)
end

RL.id(p::PlotterAlgo) = "Plotter"
RL.description(p::PlotterAlgo) = "Plots episode-rewards on y-axis vs episode-number on x-axis at the end of each episode. Every `save_interval` episodes, the plot is saved to logdir/plots/rpe.png"

function RL.init!(p::PlotterAlgo, r::RLRun)
    rm(joinpath(r.logdir, "plots"), force=true, recursive=true)
    mkpath(joinpath(r.logdir, "plots"))
end

function RL.on_env_terminal_step!(p::PlotterAlgo, r::RLRun)
    push!(p.episode_rewards, r.episode_reward)
    # plot(1:r.total_episodes, p.episode_rewards)
    pl = plot(1:r.total_episodes, p.episode_rewards, title="Reward vs Episode", label="")
    xlabel!("Episode No.")
    ylabel!("Reward")
    if r.total_episodes % p.save_interval == 0
        savefig(pl, joinpath(r.logdir, "plots", "rpe.png"))
    end
    !p.no_display && display(pl)
end

function RL.on_run_break!(p::PlotterAlgo, r::RLRun)
    savefig(joinpath(r.logdir, "plots", "rpe.png"))
end
