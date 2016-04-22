local E2 = require 'DownsampledEncoder2'
local D2 = require 'DownsampledDecoder2'

function load_pred_net(p_args)
    local enc2 = E2(p_args.dim_hidden, p_args.color_channels,
                        p_args.feature_maps)
    local dec2 = D2(p_args.dim_hidden, p_args.color_channels,
                        p_args.feature_maps, p_args.noise,
                        p_args.sharpening_rate, p_args.scheduler_iteration,
                        p_args.num_heads)

    local net = nn.Sequential()
    net:add(enc2)
    net:add(dec2)

    if opt.gpu >=0 then
        net:cuda()
    end
    collectgarbage()

    return net
end
