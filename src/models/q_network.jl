using Flux

struct QModel
    net::Chain
end
@Flux.functor QModel

"""For images, Mnih DQN network. Otherwise two hiddens layers of 64, 32 nodes."""
function QModel(state_shape::Tuple, num_actions::Integer)
    net = nothing
    if length(state_shape) == 3
        net = Chain(
            Conv((8, 8), state_shape[3] => 32, relu; stride=4),
            Conv((4, 4), 32 => 64, relu; stride=2),
            Conv((3, 3), 64 => 64, relu; stride=1),
            flatten,
            Dense(3136, 512, relu),
            Dense(512, num_actions))
    elseif length(state_shape) == 1
        net = Chain(
            Dense(state_shape[1], 64, relu),
            Dense(64, 32, relu),
            Dense(32, num_actions))
    end
    return QModel(net)
end

function (qmodel::QModel)(states::AbstractArray{Float32})
    return qmodel.net(states)
end