struct CompositeRLAlgo <: AbstractRLAlgo
    sub_algos::Array{AbstractRLAlgo}
end

function Base.push!(algo::CompositeRLAlgo, subalgo::AbstractRLAlgo)
    push!(algo.sub_algos, subalgo)
end

id(algo::CompositeRLAlgo) = join([id(sa) for sa in algo.sub_algos], " + ")

function description(algo::CompositeRLAlgo)
    contents = join(["$(id(sa)): $(description(sa))" for sa in algo.sub_algos], "\n\t")
    return "(Composite) $(id(algo)):\n\t$contents"
end

function init!(algo::CompositeRLAlgo, rlrun::RLRun)
    for sub_algo in algo.sub_algos
        init!(sub_algo, rlrun)
    end
    return nothing
end

function on_run_start!(algo::CompositeRLAlgo, rlrun::RLRun)
    for sub_algo in algo.sub_algos
        on_run_start!(sub_algo, rlrun)
    end
    return nothing
end

function on_env_reset!(algo::CompositeRLAlgo, rlrun::RLRun)
    for sub_algo in algo.sub_algos
        on_env_reset!(sub_algo, rlrun)
    end
    return nothing
end

function on_env_step!(algo::CompositeRLAlgo, rlrun::RLRun)
    for sub_algo in algo.sub_algos
        on_env_step!(sub_algo, rlrun)
    end
    return nothing
end

function on_env_terminal_step!(algo::CompositeRLAlgo, rlrun::RLRun)
    for sub_algo in algo.sub_algos
        on_env_terminal_step!(sub_algo, rlrun)
    end
    return nothing
end

function on_run_break!(algo::CompositeRLAlgo, rlrun::RLRun)
    for sub_algo in algo.sub_algos
        on_run_break!(sub_algo, rlrun)
    end
    return nothing
end

function on_env_close!(algo::CompositeRLAlgo, rlrun::RLRun)
    for sub_algo in algo.sub_algos
        on_env_close!(sub_algo, rlrun)
    end
    return nothing
end

function on_run_finish!(algo::CompositeRLAlgo, rlrun::RLRun)
    for sub_algo in algo.sub_algos
        on_run_finish!(sub_algo, rlrun)
    end
    return nothing
end

function act!(algo::CompositeRLAlgo, rlrun::RLRun)
    last_action = nothing
    for sub_algo in algo.sub_algos
        action = act!(sub_algo, rlrun)
        if !isnothing(action)
            last_action = action
        end
    end
    return last_action
end
