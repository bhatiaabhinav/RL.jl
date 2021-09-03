using ..RL
using Plots

mutable struct PlotterAlgo <: AbstractRLAlgo
    save_interval::Int
    no_display::Bool
    plot
    PlotterAlgo(;save_interval=1000, no_display=false) = new(save_interval, no_display, nothing)
end

RL.id(p::PlotterAlgo) = "Plotter"
RL.description(p::PlotterAlgo) = "Plots episode-rewards on y-axis vs episode-number on x-axis at the end of each episode. Every `save_interval` episodes, the plot is saved to logdir/plots/rpe.png"

function RL.init!(p::PlotterAlgo, r::RLRun)
    rm(joinpath(r.logdir, "plots"), force=true, recursive=true)
    mkpath(joinpath(r.logdir, "plots"))
    if p.no_display
        ENV["GKSwstype"] = "100"  # Headless mode
    end
    gr()
    p.plot = Plots.plot([], [], label="Reward", legend=:topleft)
    plot!(p.plot, [], [], label="Moving Avg")
    Plots.xlabel!(p.plot, "Episode No.")
    Plots.ylabel!(p.plot, "Reward")
end

function RL.on_env_terminal_step!(p::PlotterAlgo, r::RLRun)
    push!(p.plot, [(r.total_episodes, r.episode_reward), (r.total_episodes, r.moving_window_rpe)])
    if r.total_episodes % p.save_interval == 0
        !p.no_display  &&  display(p.plot)
        Plots.savefig(p.plot, joinpath(r.logdir, "plots", "rpe.png"))
    end
end

function RL.on_run_finish!(p::PlotterAlgo, r::RLRun)
    Plots.savefig(p.plot, joinpath(r.logdir, "plots", "rpe.png"))
end
