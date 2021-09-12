import ..RL: RLRun, AbstractRLAlgo, id, description, init!, act!, seed_action_space!, sample_action_space!

struct RandomPolicyAlgo <: AbstractRLAlgo
    RandomPolicyAlgo(;kwargs...) = new()
end

id(algo::RandomPolicyAlgo) = "RandomPolicy"

description(algo::RandomPolicyAlgo) = "Samples actions using `sample_action_space!` call. `seed_action_space!` is called during init with the provided seed."

function init!(algo::RandomPolicyAlgo, rlrun::RLRun)
    seed_action_space!(rlrun.env, rlrun.seed)
    return nothing
end

function act!(algo::RandomPolicyAlgo, rlrun::RLRun)
    return sample_action_space!(rlrun.env)
end
