--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

if not dqn then
    require "initenv"
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-framework', 'alewrap', 'name of training framework')
cmd:option('-env', '', 'name of environment to use')
cmd:option('-game_path', '../roms/', 'path to environment file (ROM)')
cmd:option('-env_params', 'useRGB=true', 'string of environment parameters')
cmd:option('-pool_frms', 'type=\"max\",size=2',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 4, 'how many times to repeat action')
cmd:option('-random_starts', 30, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,
           'saves the agent network in a separate file')
cmd:option('-prog_freq', 10000, 'frequency of progress output')
cmd:option('-save_freq', 125000, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 250000, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-steps', 50000000, 'number of training steps to perform')
cmd:option('-eval_steps', 125000, 'number of evaluation steps')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 4, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')

-- global args
cmd:option('-global_fixweights', false, 'fix encoder weights')
cmd:option('-global_reshape', false, 'if you want to encode each image separately')
cmd:option('-pretrained_path', '', 'path of pretrained network')
cmd:option('-global_lambda', 1, 'weight on grad params for dqn net')
cmd:option('-learn_freq', 5, 'how many perceives before we do a pred net pass')
cmd:option('-lr', '', 'learning rate') -- just using global lr for now



cmd:text()

local opt = cmd:parse(arg)
print(opt)

local f = io.open(opt.name .. '.log', 'w')
for key, val in pairs(opt) do
  f:write(tostring(key) .. ": " .. tostring(val) .. "\n")
end
f:flush()
f:close()

global_args = {fixweights = opt.global_fixweights,
               reshape = opt.global_reshape,
               pretrained_path = opt.pretrained_path,
               lambda = opt.global_lambda,
               learn_freq = opt.learn_freq,
               pred_lr = opt.plr}
if opt.network and not(opt.network == '') then
    opt.agent_params = opt.agent_params..',network='..opt.network
end
if opt.lr and not(opt.lr == '') then
    opt.agent_params = opt.agent_params..',lr='..opt.lr
end

--- General setup.
local game_env, game_actions, agent, opt = setup(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local learn_start = agent.learn_start
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
local step = 0
time_history[1] = 0

local total_reward
local nrewards
local nepisodes
local episode_reward

-- screen is (1 x 3 x 210 x 160)
local screen, reward, terminal = game_env:getState()

-- don't do this if not necessary
local split = true
-- if opt.agent_params.network:match('convnet') then  -- naive dqn
--     split = false
-- end


print("Iteration ..", step)
while step < opt.steps do
    step = step + 1
    local action_index = agent:perceive(reward, screen, terminal)

    -- game over? get next game!
    if not terminal then
        screen, reward, terminal = game_env:step(game_actions[action_index], true)
    else
        if opt.random_starts > 0 then
            screen, reward, terminal = game_env:nextRandomGame()
        else
            screen, reward, terminal = game_env:newGame()
        end
    end

    if step % opt.prog_freq == 0 then
        assert(step==agent.numSteps, 'trainer step: ' .. step ..
                ' & agent.numSteps: ' .. agent.numSteps)
        print("Steps: ", step)
        agent:report()
        collectgarbage()
    end

    if step%1000 == 0 then collectgarbage() end

    if step % opt.eval_freq == 0 and step > learn_start then

        screen, reward, terminal = game_env:newGame()

        total_reward = 0
        nrewards = 0
        nepisodes = 0
        episode_reward = 0

        local eval_time = sys.clock()
        for estep=1,opt.eval_steps do
            local action_index = agent:perceive(reward, screen, terminal, true, 0.05)

            -- Play game in test mode (episodes don't end when losing a life)
            screen, reward, terminal = game_env:step(game_actions[action_index])

            if estep%1000 == 0 then collectgarbage() end

            -- record every reward
            episode_reward = episode_reward + reward
            if reward ~= 0 then
               nrewards = nrewards + 1
            end

            if terminal then
                total_reward = total_reward + episode_reward
                episode_reward = 0
                nepisodes = nepisodes + 1
                screen, reward, terminal = game_env:nextRandomGame()
            end
        end

        eval_time = sys.clock() - eval_time
        start_time = start_time + eval_time
        agent:compute_validation_statistics(split)
        local ind = #reward_history+1
        total_reward = total_reward/math.max(1, nepisodes)

        if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
            agent.network:clearState()
            collectgarbage()
            agent.best_network = agent.network:clone()  -- there may be a problem here (but we just want to clone the weights right?)
        end

        if agent.v_avg then
            v_history[ind] = agent.v_avg  -- affected by validation_statistics
            td_history[ind] = agent.tderr_avg  -- affected by validation_statistics
            qmax_history[ind] = agent.q_max
        end
        print("V", v_history[ind], "TD error", td_history[ind], "Qmax", qmax_history[ind])

        reward_history[ind] = total_reward
        reward_counts[ind] = nrewards
        episode_counts[ind] = nepisodes

        time_history[ind+1] = sys.clock() - start_time

        local time_dif = time_history[ind+1] - time_history[ind]

        local training_rate = opt.actrep*opt.eval_freq/time_dif

        print(string.format(
            '\nSteps: %d (frames: %d), reward: %.2f, epsilon: %.2f, lr: %G, ' ..
            'training time: %ds, training rate: %dfps, testing time: %ds, ' ..
            'testing rate: %dfps,  num. ep.: %d,  num. rewards: %d',
            step, step*opt.actrep, total_reward, agent.ep, agent.lr, time_dif,
            training_rate, eval_time, opt.actrep*opt.eval_steps/eval_time,
            nepisodes, nrewards))
    end

    if step % opt.save_freq == 0 or step == opt.steps then
        local s, a, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_r,
            agent.valid_s2, agent.valid_term
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = nil, nil, nil, nil, nil, nil, nil


        local enc_w, enc_dw, enc_g, enc_g2, enc_delta, enc_delta2, enc_deltas,
            enc_tmp = agent.enc_w, agent.enc_dw, agent.enc_g, agent.enc_g2,
            agent.enc_delta, agent.enc_delta2, agent.enc_deltas, agent.enc_tmp
        agent.enc_w, agent.enc_dw, agent.enc_g, agent.enc_g2, agent.enc_delta,
            agent.enc_delta2, agent.enc_deltas, agent.enc_tmp = nil, nil, nil,
            nil, nil, nil, nil, nil

        local dec_w, dec_dw, dec_g, dec_g2, dec_delta, dec_delta2, dec_deltas,
            dec_tmp = agent.dec_w, agent.dec_dw, agent.dec_g, agent.dec_g2,
            agent.dec_delta, agent.dec_delta2, agent.dec_deltas, agent.dec_tmp
        agent.dec_w, agent.dec_dw, agent.dec_g, agent.dec_g2, agent.dec_delta,
            agent.dec_delta2, agent.dec_deltas, agent.dec_tmp = nil, nil, nil,
            nil, nil, nil, nil, nil


        local filename = opt.name
        if opt.save_versions > 0 then
            filename = filename .. "_" .. math.floor(step / opt.save_versions)
        end
        filename = filename

        agent.network:clearState()
        -- print(agent.best_network)
        if agent.best_network then
            agent.best_network:clearState()
        end
        collectgarbage()
        torch.save(filename .. ".t7", {agent = agent,
                                model = agent.network,
                                best_model = agent.best_network,
                                reward_history = reward_history,
                                reward_counts = reward_counts,
                                episode_counts = episode_counts,
                                time_history = time_history,
                                v_history = v_history,
                                td_history = td_history,
                                qmax_history = qmax_history,
                                arguments=opt})
        if opt.saveNetworkParams then
            local nets = {network=w:clone():float()}
            torch.save(filename..'.params.t7', nets, 'ascii')
        end
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = s, a, r, s2, term


        agent.enc_w, agent.enc_dw, agent.enc_g, agent.enc_g2, agent.enc_delta,
            agent.enc_delta2, agent.enc_deltas, agent.enc_tmp = enc_w, enc_dw,
            enc_g, enc_g2, enc_delta, enc_delta2, enc_deltas, enc_tmp
        agent.dec_w, agent.dec_dw, agent.dec_g, agent.dec_g2, agent.dec_delta,
            agent.dec_delta2, agent.dec_deltas, agent.dec_tmp = dec_w, dec_dw,
            dec_g, dec_g2, dec_delta, dec_delta2, dec_deltas, dec_tmp


        print('Saved:', filename .. '.t7')
        io.flush()
        collectgarbage()
    end
end
