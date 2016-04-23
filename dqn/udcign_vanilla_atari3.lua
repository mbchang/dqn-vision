require 'udcign'
return function(args)
    local cp_path = '/home/mbchang/code/unsupervised-dcign/logslink/'..
                    'atari_vanilla_motion_scale_3_noise_0.1_heads_3_'..
                    'sharpening_rate_10_gpu_learning_rate_0.0002_'..
                    'dataset_name_breakout_frame_interval_3/'
    args.vanilla = true
    return load_udcign_encoder(cp_path, args)
end
