-- udcign

require "initenv"
local optnet = require 'optnet'

function load_encoder(encoder, args)
    local net = nn.Sequential()
    local n_hidden = 128
    local encoder_dim = args.hist_len*args.p_dim_hidden

    net:add(encoder)

    local decoder = nn.Sequential()
    decoder:add(nn.Linear(encoder_dim, n_hidden))
    decoder:add(nn.ReLU())
    decoder:add(nn.Linear(n_hidden, n_hidden))
    decoder:add(nn.ReLU())
    decoder:add(nn.Linear(n_hidden, args.n_actions))
    net:add(decoder)


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
