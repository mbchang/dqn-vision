-- udcign

require "initenv"

-- these modules don't actually matter for the encoder, but they just matter
-- for loading the model in
require 'Noise'
require 'ScheduledWeightSharpener'
require 'ChangeLimiter'
local Encoder = require 'AtariEncoder_vanilla'

function load_udcign_encoder(cp_path, args)
    local checkpoint = torch.load(cp_path)
    local opt = checkpoint.opt
    print(opt)
    local model = checkpoint.model

    -- (bsize x 3 x 210 x 160) to (bsize x 200)
    local encoder = model.modules[1].modules[2]
    print(encoder)
    local encoder_dim = args.hist_len*200 -- hardcoded

    -- input is (bsize x 3 x 210 x 160)
    local net = nn.Sequential()

    -- encode each of the histlen images separately
    net:add(nn.Reshape(args.hist_len,3,210,160))

    -- net:add(nn.Transpose({1,2}))
    -- should I transpose here?

    net:add(nn.SplitTable(2))

    -- split on first dimension, to get 4 tensors of (bsize, 3,210,160)
    -- join on first dimension: hist_len*200
    local vision = nn.ParallelTable()
    vision:add(encoder)
    --
    -- -- share weights. Number of copies = args.hist_len
    for i = 2,args.hist_len do
        -- I have to clone this
        -- (dim_hidden, color_channels, feature_maps, batch_norm)
        local enc_copy = Encoder(200, 3, 72, 0.1)
        enc_copy:cuda()
        print(enc_copy)
        -- local enc_copy =  model.modules[1].modules[2]:clone()  -- is there a way to just get the architecture?
        enc_copy:share(encoder,'weight', 'bias', 'gradWeight', 'gradBias')
        vision:add(enc_copy)
        -- vision:add(encoder:clone('weight', 'bias', 'gradWeight', 'gradBias'))
    end

    net:add(vision) -- output is table of 200 dim vectors
    net:add(nn.JoinTable(2))

    -- apparently this outputs a table (16, 200); has something to do with batch mode

    -- add the last fully connected layer (to actions)
    net:add(nn.Linear(encoder_dim, args.n_actions))

    if args.gpu >=0 then
        net:cuda()
    end
    if args.verbose >= 2 then
        print(net)
        -- print('Convolutional layers flattened output size:', nel)
    end
    cutorch.synchronize()
    collectgarbage()
    return net
end
