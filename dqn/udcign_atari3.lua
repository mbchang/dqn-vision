require 'udcign'
return function(args)
    local cp_path = '/om/user/wwhitney/unsupervised-dcign/networks/'..
                    'atari_motion_scale_3_noise_0.1_heads_3_sharpening_rate_10_'..
                    'gpu_learning_rate_0.0002_dataset_name_breakout_'..
                    'frame_interval_3/epoch28.12_0.2033.t7'
    args.vanilla = false
    return load_udcign_encoder(cp_path, args)
end
