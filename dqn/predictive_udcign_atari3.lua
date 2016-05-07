local E2 = require 'DownsampledEncoder2_parallel'
local D2 = require 'DownsampledDecoder2'

function load_pred_net(p_args)
    local enc2 = E2(p_args.p_dim_hidden, p_args.p_color_channels,
                        p_args.p_feature_maps)
    local dec2 = D2(p_args.p_dim_hidden, p_args.p_color_channels,
                        p_args.p_feature_maps, p_args.p_noise,
                        p_args.p_sharpening_rate, p_scheduler_iteration,  -- global
                        p_args.p_num_heads)

    local net = nn.Sequential()
    net:add(enc2)
    net:add(dec2)

    if p_args.gpu >=0 then
        net:cuda()
    end
    collectgarbage()

    return net
end
