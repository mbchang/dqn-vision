require 'load_udcign'

return function(args)
    -- local cp_path = '/om/user/wwhitney/unsupervised-dcign/networks/down_motion_scale_3_noise_0.1_heads_3_sharpening_rate_10_gpu_learning_rate_0.0001_model_autoencoder_dataset_name_space_invaders_frame_interval_1/epoch50.00_0.1298.t7'
    local cp_path = global_args.pretrained_path
    print('cp path:'..cp_path)
    args.vanilla = true
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
