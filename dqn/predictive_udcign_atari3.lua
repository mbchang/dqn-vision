local E2 = require 'DownsampledEncoder2'
local D2 = require 'DownsampledDecoder2'

function load_pred_net(args)
    local dim_hidden = 200
    local color_channels = 1
    local feature_maps = 4
    local noise = 0.1
    local sharpening_rate = 10
    local scheduler_iteration = torch.zeros(1)  -- TODO! scheduler_iteration should be a field in NeuralQPredictiveLearner
    local num_heads = 1

    local enc2 = E2(dim_hidden, color_channels, feature_maps)
    local dec2 = D2(dim_hidden, color_channels, feature_maps, noise, sharpening_rate, args.scheduler_iteration, num_heads)

    local net = nn.Sequential()
    net:add(enc2)
    net:add(dec2)

    if opt.gpu >=0 then
        net:cuda()
    end
    collectgarbage()

    return net
end
