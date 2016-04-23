-- udcign

require "initenv"
local optnet = require 'optnet'

function load_encoder(encoder, args)
    local encoder_dim = args.hist_len*args.p_dim_hidden

    local net = nn.Sequential()

    -- encoder
    local pre_encoder = nn.Sequential()
    pre_encoder:add(nn.Reshape(args.minibatch_size*args.hist_len,1,84,84))
    pre_encoder:add(encoder) -- output is (bsize*hist_len, 200)
    net:add(pre_encoder)

    -- add the last fully connected layer (to actions)
    local linear = nn.Sequential()
    linear:add(nn.Reshape(args.minibatch_size,encoder_dim))
    linear:add(nn.Linear(encoder_dim, args.n_actions))

    net:add(linear)

    if args.gpu >=0 then
        net:cuda()
        cutorch.synchronize()
    end
    if args.verbose >= 2 then
        print(net)
    end
    collectgarbage()
    return net
end

-- code to test
-- args = {}
-- args.hist_len = 4
-- args.p_dim_hidden = 200
-- args.minibatch_size = 32
-- args.n_actions = 4
-- args.p_color_channels = 1
-- args.p_feature_maps = 72
-- args.gpu = -1
-- args.verbose = 1
--
-- local E2 = require 'DownsampledEncoder2'
-- local enc = E2(args.p_dim_hidden, args.p_color_channels,
--                     args.p_feature_maps)
--
-- local net = load_udcign_untrained(enc,args)
--
-- local batch = torch.rand(32,4,84,84)
-- print(net)
-- local out = net:forward(batch)
-- print(out:size())
