require 'load_udcign'
require 'Print'
require 'nn'
require 'nngraph'
require 'Print'
require 'ChangeLimiter'
require 'Noise'
require 'ScheduledWeightSharpener'


return function(args)
    local cp_path = '/om/user/wwhitney/unsupervised-dcign/networks/down_motion_scale_3_noise_0.1_heads_3_sharpening_rate_10_gpu_learning_rate_0.0002_model_disentangled_dataset_name_space_invaders_frame_interval_1/epoch50.00_0.1369.t7'
    args.vanilla = false
    args.p_dim_hidden = 200

    local checkpoint = torch.load(cp_path)

    local opt = checkpoint.opt
    local model = checkpoint.model

    local encoder
    if args.vanilla then
        encoder = model.modules[1]
    else
        encoder = model.modules[1].modules[2]
    end

    encoder:clearState()  -- clear output and gradInput
    collectgarbage()

    return load_encoder(encoder, args)
end
