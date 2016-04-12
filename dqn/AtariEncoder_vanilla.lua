require 'nn'
require 'nngraph'

require 'Print'


local AtariEncoder = function(dim_hidden, color_channels, feature_maps)

    local filter_size = 5
    -- local inputs = {
    --         nn.Identity()():annotate{name="input1"},
    --         nn.Identity()():annotate{name="input2"},
    --     }
    -- local input = nn.Identity()()

    -- make two copies of an encoder

    local enc1 = nn.Sequential()
    enc1:add(nn.SpatialConvolution(color_channels, feature_maps, filter_size, filter_size))
    enc1:add(nn.SpatialMaxPooling(2,2,2,2))
    -- if batch_norm then
    --     enc1:add(nn.SpatialBatchNormalization(feature_maps))
    -- end
    enc1:add(nn.Threshold(0,1e-6))

    enc1:add(nn.SpatialConvolution(feature_maps, feature_maps/2, filter_size, filter_size))
    enc1:add(nn.SpatialMaxPooling(2,2,2,2))
    -- if batch_norm then
    --     enc1:add(nn.SpatialBatchNormalization(feature_maps/2))
    -- end
    enc1:add(nn.Threshold(0,1e-6))

    enc1:add(nn.SpatialConvolution(feature_maps/2, feature_maps/4, filter_size, filter_size))
    enc1:add(nn.SpatialMaxPooling(2,2,2,2))
    -- if batch_norm then
    --     enc1:add(nn.SpatialBatchNormalization(feature_maps/4))
    -- end
    enc1:add(nn.Threshold(0,1e-6))

    enc1:add(nn.Reshape((feature_maps/4) * 22*16))
    enc1:add(nn.Linear((feature_maps/4) * 22*16, dim_hidden))

    -- enc1 = enc1(input)  -- if I just do enc1 on the inputs[2], then it's as if I'm just doing regular autoencoder!
    -- return nn.gModule({input}, {enc1})
    return enc1
end

return AtariEncoder
