require 'udcign'
return function(args)
    -- local cp_path = '/om/user/wwhitney/unsupervised-dcign/networks/'..
    --                 'atari_motion_scale_3_noise_0.1_heads_3_sharpening_rate_10_'..
    --                 'gpu_learning_rate_0.0002_dataset_name_breakout_'..
    --                 'frame_interval_3/epoch28.12_0.2033.t7'
    local cp_path = '/om/user/wwhitney/unsupervised-dcign/networks/down_motion_scale_3_noise_0.1_heads_3_sharpening_rate_10_gpu_learning_rate_0.0001_model_autoencoder_dataset_name_space_invaders_frame_interval_1/epoch50.00_0.1298.t7'
    args.vanilla = true
    return load_udcign_encoder(cp_path, args)
end
