local E = require 'DownsampledEncoder2'
local D = require 'DownsampledDecoder'

function load_pred_net(p_args)
    local enc = E(p_args.p_dim_hidden, p_args.p_color_channels,
                        p_args.p_feature_maps)
    local dec = D(p_args.p_dim_hidden, p_args.p_color_channels,
                        p_args.p_feature_maps)

    local net = nn.Sequential()
    net:add(enc)
    net:add(dec)

    if p_args.gpu >=0 then
        net:cuda()
    end
    collectgarbage()

    return net
end
