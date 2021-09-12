using ..RL
using Plots 
using UnicodePlots

mutable struct PlotInTerminalAlgo <: AbstractRLAlgo
    interval::Int
    plot
    PlotInTerminalAlgo(;interval=100) = new(interval, nothing)
end

RL.id(p::PlotInTerminalAlgo) = "In-Terminal Plotter"
RL.description(p::PlotInTerminalAlgo) = "Plots in the terminal the episode-rewards on y-axis vs episode-number on x-axis, every `interval` episodes"

function RL.init!(p::PlotInTerminalAlgo, r::RLRun)
    unicodeplots()
    p.plot = Plots.plot([], [], label="Reward")
    plot!(p.plot, [], [], label="Moving Avg")
    Plots.xlabel!(p.plot, "Episode No.")
    Plots.ylabel!(p.plot, "Reward")
    return nothing
end

function RL.on_env_terminal_step!(p::PlotInTerminalAlgo, r::RLRun)
    push!(p.plot, [(r.total_episodes, r.episode_reward), (r.total_episodes, r.moving_window_rpe)])
    if r.total_episodes % p.interval == 0
        display(p.plot)
        println()
    end
    return nothing
end

function RL.on_run_finish!(p::PlotInTerminalAlgo, r::RLRun)
    display(p.plot)
    println()
    return nothing
end
