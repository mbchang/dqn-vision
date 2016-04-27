import os
import sys
import pprint

dry_run = '--dry-run' in sys.argv
local   = '--local' in sys.argv
detach  = '--detach' in sys.argv

# local = True
# dry_run = True
# detatch = True

if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")

if not os.path.exists("slurm_scripts"):
    os.makedirs("slurm_scripts")

networks_prefix = "../networks/"

# network (netfile), agent, seed, gpu, agent_params contains network
# agent_params="network="$netfile" -global_fixweights

# agent --> NeuralQLearner or NeuralQLearnerReshape
# network --> netfile
# name --> savefile

seeds = range(1,3)
envs = ['breakout','space_invaders']
agents = ['NeuralQLearner', 'NeuralQLearnerReshape']
networks = ['\"vanilla_trained_atari3\"','\"udcign_trained_atari3\"']


myjobs = []   # seed, env, learn, agent, network
i = 0
for seed in seeds:
    for env in envs:
        for agent in agents:

            if agent == 'NeuralQLearner':
                job = {'seed':seed,'env':env,'agent':agent}
                job['network'] = '\"convnet_atari3\"'
                myjobs.append(job)
            else:
                for network in networks:
                    for learn in [True, False]:
                        job = {'seed':seed,'env':env,'agent':agent,'network':network,'learn':learn}

                        if env == 'breakout':
                            if network == '\"vanilla_trained_atari3\"':
                                pass
                            elif network == '\"udcign_trained_atari3\"':
                                pass
                        elif env == 'space_invaders':
                            if network == '\"vanilla_trained_atari3\"':
                                job['pretrained_path'] = '/om/user/wwhitney/unsupervised-dcign/networks/down_motion_scale_3_noise_0.1_heads_3_sharpening_rate_10_gpu_learning_rate_0.0001_model_autoencoder_dataset_name_space_invaders_frame_interval_1/epoch50.00_0.1298.t7'
                            elif network == '\"udcign_trained_atari3\"':
                                job['pretrained_path'] = '/om/user/wwhitney/unsupervised-dcign/networks/down_motion_scale_3_noise_0.1_heads_3_sharpening_rate_10_gpu_learning_rate_0.0002_model_disentangled_dataset_name_space_invaders_frame_interval_1/epoch50.00_0.1369.t7'

                        myjobs.append(job)


if dry_run:
    print "NOT starting jobs:"
else:
    print "Starting jobs:"
for job in myjobs:
    assert 'seed' in job
    assert 'env' in job
    assert 'agent' in job
    assert 'network' in job
    if job['agent'] != 'NeuralQLearner' and job['env'] != 'breakout':
        assert 'pretrained_path' in job
        assert 'learn' in job

    # create job string
    flagstring = ""
    jobname = ""
    for flag in job:
        if flag == 'learn':
            if not job['learn']:  # we want to fix weights
                flagstring += ' -global_fixweights'
            else:
                jobname += '_learn'
        elif isinstance(job[flag], bool):
            if job[flag]:
                jobname = jobname + "_" + flag
                flagstring = flagstring + " -" + flag
            else:
                print "WARNING: Excluding 'False' flag " + flag
        else:
            flagstring = flagstring + " -" + flag + " " + str(job[flag])
            if flag == 'network':
                if job[flag] == '\"udcign_trained_atari3\"':
                    jobname += '_disentangled'
                elif job[flag] == '\"vanilla_trained_atari3\"':
                    jobname += '_vanilla'
                elif job[flag] == '\"convnet_atari3\"':
                    jobname += '_dqn'
                else:
                    assert False, 'unknown netfile: ' + job[flag]
            if flag == 'agent':
                if job[flag] == 'NeuralQLearner':
                    jobname += '_naive'
                elif job[flag] == 'NeuralQLearnerReshape':
                    jobname += '_offline'
                    flagstring += ' -global_reshape'
                elif job[flag] == 'NeuralQLearnerPredictive':
                    assert False, "Did you implement this yet?"
                    jobname += '_online'
                    flagstring += ' -global_reshape'
                else:
                    assert False,'unknown agent'
            else:
                if flag != 'pretrained_path':
                    jobname = jobname + "_" + flag + "_" + str(job[flag])
    jobname = jobname[1:]
    flagstring = flagstring[1:]

    # construct import file
    import_string = "-name " + networks_prefix + jobname

    jobcommand = "th train_agent.lua $args {flag_string} {import_string}".format(
        flag_string = flagstring,
        import_string = import_string)


    script_path = 'run_gpu_' + jobname
    os.system('cp run_gpu_template ' + script_path)

    with open(script_path.replace('\"',''), 'a') as script_file:
        script_file.write('\n')
        script_file.write(jobcommand)
        script_file.write('\n')

    if local:
        if not dry_run:
            if detach:
                os.system('bash ' + script_path + ' 2> slurm_logs/' + jobname + '.err 1> slurm_logs/' + jobname + '.out &')
            else:
                os.system('bash ' + script_path)

    else:
        with open('slurm_scripts/' + jobname + '.slurm', 'w') as slurmfile:
            slurmfile.write("#!/bin/bash\n")
            slurmfile.write("#SBATCH --job-name"+"=" + jobname + "\n")
            slurmfile.write("#SBATCH --output=slurm_logs/" + jobname + ".out\n")
            slurmfile.write("#SBATCH --error=slurm_logs/" + jobname + ".err\n")
            slurmfile.write('bash ' + script_path)

        if not dry_run:
            os.system("sbatch -N 1 -c 1 --gres=gpu:1 -p gpu --mem=16000 --time=6-23:00:00 slurm_scripts/" + jobname + ".slurm &")
