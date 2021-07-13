module Algos

export RandomPolicyAlgo, RenderAlgo, DQNAlgo, PlotterAlgo, WandbLoggerAlgo

include("random.jl")
include("render.jl")
include("dqn.jl")
include("plotter.jl")
include("wandb_logger.jl")

end  # module