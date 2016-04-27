local E2 = require 'DownsampledEncoder2'
require 'load_udcign'

return function(args)
    args.p_dim_hidden = 200
    args.p_color_channels = 1
    args.p_feature_maps = 72
    local enc = E2(args.p_dim_hidden, args.p_color_channels,
                        args.p_feature_maps)
    return load_encoder(enc, args)
end
