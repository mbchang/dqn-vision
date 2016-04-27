require 'udcign'
return function(args)
    local cp_path = '/om/user/wwhitney/unsupervised-dcign/networks/down_motion_scale_3_noise_0.1_heads_3_sharpening_rate_10_gpu_learning_rate_0.0002_model_disentangled_dataset_name_space_invaders_frame_interval_1/epoch50.00_0.1369.t7'
    args.vanilla = false
    return load_udcign_encoder(cp_path, args)
end
