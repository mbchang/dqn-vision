local E2 = require 'DownsampledEncoder2'
local D2 = require 'DownsampledDecoder2'

function load_pred_net(iteration_container)
    local dim_hidden = 200
    local color_channels = 1
    local feature_maps = 72
    local noise = 0.1
    local sharpening_rate = 10
    local num_heads = 3

    local enc2 = E2(dim_hidden, color_channels, feature_maps)
    local dec2 = D2(dim_hidden, color_channels, feature_maps, noise, sharpening_rate, iteration_container, num_heads)

    local net = nn.Sequential()
    net:add(enc2)
    net:add(dec2)

    if opt.gpu >=0 then
        net:cuda()
    end
    collectgarbage()

    return net
end
