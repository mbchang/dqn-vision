-- udcign

require "initenv"

-- these modules don't actually matter for the encoder, but they just matter
-- for loading the model in
require 'Noise'
require 'ScheduledWeightSharpener'
require 'ChangeLimiter'

function load_udcign_encoder(cp_path, args)
    local checkpoint = torch.load(cp_path)
    local opt = checkpoint.opt
    local model = checkpoint.model

    -- (bsize x 3 x 210 x 160) to (bsize x 200)
    local encoder = model.modules[1].modules[2]
    -- encoder:cuda()
    local encoder_dim = args.hist_len*200 -- hardcoded
    --
    -- local input = nn.Identity()()
    -- local reshaped = nn.Reshape(args.hist_len,3,210,160)(input)
    -- local splitted = nn.SplitTable(1)(reshaped)
    --
    -- local encoder_clones = {encoder}
    -- for _ = 2,args.hist_len do
    --     local enc_copy =  model.modules[1].modules[2]:clone()
    --     enc_copy:share(encoder,'weight', 'bias', 'gradWeight', 'gradBias')
    --     table.insert(encoder_clones, enc_copy)
    -- end
    --
    -- local encodings = {}
    -- for i = 1, args.hist_len do
    --     table.insert(encodings,
    --         encoder_clones[i](splitted[i]))
    -- end
    --
    -- local joined = nn.JoinTable(1)(encodings)
    -- local output = nn.Linear(encoder_dim, args.n_actions)(joined)
    --
    -- local gmodule = nn.gModule(input, output)
    --
    -- if args.gpu >=0 then
    --     gmodule:cuda()
    -- end
    -- return gmodule

    --
    -- -- input is (bsize x 3 x 210 x 160)
    local net = nn.Sequential()
    --
    -- encode each of the histlen images separately
    net:add(nn.Reshape(args.hist_len,3,210,160))

    net:add(nn.SplitTable(1))

    -- split on first dimension, to get 4 tensors of (3,210,160)
    -- join on first dimension: hist_len*200
    local vision = nn.ParallelTable()
    vision:add(encoder)

    -- share weights. Number of copies = args.hist_len
    for i = 2,args.hist_len do
        local enc_copy =  model.modules[1].modules[2]:clone()  -- I have to clone this though
        enc_copy:share(encoder,'weight', 'bias', 'gradWeight', 'gradBias')
        collectgarbage()
        collectgarbage()
        vision:add(enc_copy)
        collectgarbage()
        collectgarbage()
    end

    net:add(vision) -- output is table of 200 dim vectors
    net:add(nn.JoinTable(1))

    -- add the last fully connected layer (to actions)
    net:add(nn.Linear(encoder_dim, args.n_actions))

    if args.gpu >=0 then
        net:cuda()
    end
    if args.verbose >= 2 then
        print(net)
        -- print('Convolutional layers flattened output size:', nel)
    end
    return net
end
--
-- function create_network(args)
--
--     local net = nn.Sequential()
--     net:add(nn.Reshape(unpack(args.input_dims)))  -- this has been rescaled
--
--     --- first convolutional layer
--     local convLayer = nn.SpatialConvolution
--
--     -- if args.gpu >= 0 then
--     --     --net:add(nn.Transpose({1,2},{2,3},{3,4}))  -- Michael changed this
--     --     convLayer = nn.SpatialConvolution--CUDA  -- Michael changed this
--     -- end
--
--     net:add(convLayer(args.hist_len*args.ncols, args.n_units[1],
--                         args.filter_size[1], args.filter_size[1],
--                         args.filter_stride[1], args.filter_stride[1],1))
--     net:add(args.nl())
--
--     -- Add convolutional layers
--     for i=1,(#args.n_units-1) do
--         -- second convolutional layer
--         net:add(convLayer(args.n_units[i], args.n_units[i+1],
--                             args.filter_size[i+1], args.filter_size[i+1],
--                             args.filter_stride[i+1], args.filter_stride[i+1]))
--         net:add(args.nl())
--     end
--
--     local nel
--     if args.gpu >= 0 then
--         -- net:add(nn.Transpose({4,3},{3,2},{2,1}))
--         nel = net:cuda():forward(torch.zeros(1,unpack(args.input_dims))
--                 :cuda()):nElement()
--     else
--         nel = net:forward(torch.zeros(1,unpack(args.input_dims))):nElement()
--     end
--
--     -- reshape all feature planes into a vector per example
--     net:add(nn.Reshape(nel))
--
--     -- insert your network here
--
--     -- fully connected layer
--     net:add(nn.Linear(nel, args.n_hid[1]))
--     net:add(args.nl())
--     local last_layer_size = args.n_hid[1]
--
--     for i=1,(#args.n_hid-1) do
--         -- add Linear layer
--         last_layer_size = args.n_hid[i+1]
--         net:add(nn.Linear(args.n_hid[i], last_layer_size))
--         net:add(args.nl())
--     end
--
--     -- add the last fully connected layer (to actions)
--     net:add(nn.Linear(last_layer_size, args.n_actions))
--
--     if args.gpu >=0 then
--         net:cuda()
--     end
--     if args.verbose >= 2 then
--         print(net)
--         print('Convolutional layers flattened output size:', nel)
--     end
--     return net
-- end
