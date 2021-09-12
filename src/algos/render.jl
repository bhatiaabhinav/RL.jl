import ..RL: RLRun, AbstractRLAlgo, id, description, on_env_reset!, on_env_step!, seed_action_space!, sample_action_space!, render

struct RenderAlgo <: AbstractRLAlgo
    RenderAlgo(;kwargs...) = new()
end

id(algo::RenderAlgo) = "Renderer"

description(algo::RenderAlgo) = "Invokes `render(env)` on reset and on step."

function on_env_reset!(algo::RenderAlgo, rlrun::RLRun)
    render(rlrun.env)
    return nothing
end

function on_env_step!(algo::RenderAlgo, rlrun::RLRun)
    render(rlrun.env)
    return nothing
end
