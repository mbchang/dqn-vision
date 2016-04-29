require 'nn'
require 'nngraph'

require 'Print'

-- input: {input1, input2}
-- output: {enc1, enc2}
local DownsampledEncoder2_parallel = function(dim_hidden, color_channels, feature_maps)

    local filter_size = 5
    local stride = 1
    local padding = 2
    local encoded_size = 10

    local enc1 = nn.Sequential()
    enc1:add(nn.SpatialConvolution(color_channels, feature_maps, filter_size, filter_size, stride, stride, padding, padding))
    enc1:add(nn.SpatialMaxPooling(2,2,2,2))
    enc1:add(nn.Threshold(0,1e-6))

    enc1:add(nn.SpatialConvolution(feature_maps, feature_maps/2, filter_size, filter_size, stride, stride, padding, padding))
    enc1:add(nn.SpatialMaxPooling(2,2,2,2))
    enc1:add(nn.Threshold(0,1e-6))

    enc1:add(nn.SpatialConvolution(feature_maps/2, feature_maps/4, filter_size, filter_size, stride, stride, padding, padding))
    enc1:add(nn.SpatialMaxPooling(2,2,2,2))
    enc1:add(nn.Threshold(0,1e-6))

    enc1:add(nn.Reshape((feature_maps/4) * encoded_size * encoded_size))
    enc1:add(nn.Linear((feature_maps/4) * encoded_size * encoded_size, dim_hidden))

    local enc2 = enc1:clone('weight', 'bias', 'gradWeight', 'gradBias')

    -- make two copies of an encoder
    local net = nn.ParallelTable()
    net:add(enc1)
    net:add(enc2)

    return net
end

return DownsampledEncoder2_parallel
