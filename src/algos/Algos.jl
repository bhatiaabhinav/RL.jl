module Algos

export RandomPolicyAlgo, RenderAlgo, DQNAlgo, PlotterAlgo, WandbLoggerAlgo, PlotInTerminalAlgo

include("random.jl")
include("render.jl")
include("dqn.jl")
include("plotter.jl")
include("plot_in_terminal.jl")
include("wandb_logger.jl")

end  # module