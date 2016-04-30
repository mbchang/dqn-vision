--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]


local optnet = require 'optnet'
require 'rmsprop'
require 'MotionBCECriterion'
require 'predictive_udcign_atari3'

if not dqn then
    require 'initenv'
end

local nql = torch.class('dqn.NeuralQPredictiveLearner')

-- false for backpropagating all the way
-- true for backpropagating only through the linear
local fix_pre_encoder = global_args.fixweights
local udcign_reshape = global_args.reshape


-- for adjusting lambda
local dw_ratio = {0,0}

function nql:__init(args)
    self.state_dim  = args.state_dim or 7056 -- State dimensionality.
    self.actions    = args.actions
    self.n_actions  = #self.actions
    self.verbose    = args.verbose
    self.best       = args.best

    --- epsilon annealing
    self.ep_start   = args.ep or 1
    self.ep         = 1 -- self.ep_start -- Exploration probability.
    self.ep_end     = 0.1 or self.ep
    self.ep_endt    = args.ep_endt or 1000000

    ---- learning rate annealing
    self.lr_start       = args.lr or 0.00025 --Learning rate.
    self.lr             = self.lr_start
    self.lr_end         = args.lr_end or self.lr
    self.lr_endt        = args.lr_endt or 1000000
    self.wc             = args.wc or 0  -- L2 weight cost.
    self.minibatch_size = args.minibatch_size or 32
    self.valid_size     = args.valid_size or 500

    --- Q-learning parameters
    self.discount       = args.discount or 0.99 --Discount factor.
    self.update_freq    = args.update_freq or 4
    -- Number of points to replay per learning step.
    self.n_replay       = args.n_replay or 1
    -- Number of steps after which learning starts.
    self.learn_start    = args.learn_start or 50000
     -- Size of the transition table.
    self.replay_memory  = args.replay_memory or 1000000
    self.hist_len       = args.hist_len or 4
    self.rescale_r      = args.rescale_r or 1
    self.max_reward     = args.max_reward or 1
    self.min_reward     = args.min_reward or -1
    self.clip_delta     = args.clip_delta or 1
    self.target_q       = args.target_q or 10000
    self.bestq          = 0

    self.gpu            = args.gpu

    self.ncols          = args.ncols or 1  -- number of color channels in input
    self.input_dims     = args.input_dims or {self.hist_len*self.ncols, 84, 84}  -- this incorporates hist_len!
    self.preproc        = args.preproc or "net_downsample_2x_full_y"-- name of preprocessing network
    self.histType       = args.histType or "linear"  -- history type to use
    self.histSpacing    = args.histSpacing or 1
    self.nonTermProb    = args.nonTermProb or 1
    self.bufferSize     = args.bufferSize or 512

    self.transition_params = args.transition_params or {}

    -- this is just the filename
    self.network    = args.network or self:createNetwork() -- args.network is loaded by run_gpu as the convnet_atari3

    -- check whether there is a network file
    local network_function
    if not (type(self.network) == 'string') then
        error("The type of the network provided in NeuralQLearner" ..
              " is not a string!")
    end

    -- try to load the filename
    local msg, err = pcall(require, self.network)
    if not msg then
        -- try to load saved agent
        local err_msg, exp = pcall(torch.load, self.network)
        if not err_msg then
            error("Could not find network file ")
        end
        if self.best and exp.best_model then
            self.network = exp.best_model
        else
            self.network = exp.model
        end
    else
        print('Creating Agent Network from ' .. self.network)  -- given by convnet_atari3: you should change this.
        self.network = err
        self.network = self:network()
    end

    ----------------------------------------------------------------------------
    -- training parameters for self.pred_net
    self.p_motion_scale = 3
    self.p_grad_clip = 3
    self.p_L2 = 0
    self.p_learning_rate = 0.0001  -- TODO CHANGE ME
    self.p_learning_rate_decay = 0.97
    self.p_learning_rate_decay_interval = 4000
    self.p_learning_rate_decay_after = 18000
    self.p_decay_rate = 0.95  -- rmsprop alpha
    self.enc_optim_state = {learningRate=self.p_learning_rate,
                        alpha=self.p_decay_rate}
    self.dec_optim_state = {learningRate=self.p_learning_rate,
                        alpha=self.p_decay_rate}
    self.p_lambda = global_args.lambda

    self.predictive_iteration = 0

    self.p_args = {}
    self.p_args.p_sharpening_rate = 10
    self.p_args.p_scheduler_iteration = torch.zeros(1)
    self.p_args.p_dim_hidden = 200
    self.p_args.p_color_channels = self.ncols
    self.p_args.p_feature_maps = 72
    self.p_args.p_noise = 0.1
    self.p_args.p_num_heads = 3
    self.p_args.gpu = self.gpu

    -- here, iniitialize predictive network
    self.pred_net = load_pred_net(self.p_args) -- this may be faster
    self.pred_criterion = nn.MotionBCECriterion(self.p_motion_scale)
    print('Prediction Network')
    print(self.pred_net)
    ----------------------------------------------------------------------------

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
        self.pred_net:cuda()
        self.pred_criterion:cuda()
    else
        self.network:float()
        self.pred_net:float()
        self.pred_criterion:float()
    end

    -- Load preprocessing network.
    if not (type(self.preproc == 'string')) then
        error('The preprocessing is not a string')
    end
    msg, err = pcall(require, self.preproc)
    if not msg then
        error("Error loading preprocessing net")
    end
    self.preproc = err
    self.preproc = self:preproc()
    self.preproc:float()

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
        self.pred_net:cuda()
        self.pred_criterion:cuda()
        self.tensor_type = torch.CudaTensor
    else
        self.network:float()
        self.pred_net:float()
        self.pred_criterion:float()
        self.tensor_type = torch.FloatTensor
    end

    -- Create transition table.
    ---- assuming the transition table always gets floating point input
    ---- (Foat or Cuda tensors) and always returns one of the two, as required
    ---- internally it always uses ByteTensors for states, scaling and
    ---- converting accordingly
    local transition_args = {
        stateDim = self.state_dim, numActions = self.n_actions,
        histLen = self.hist_len, gpu = self.gpu,
        maxSize = self.replay_memory, histType = self.histType,
        histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
        bufferSize = self.bufferSize
    }

    self.transitions = dqn.TransitionTable(transition_args)

    self.numSteps = 0 -- Number of perceived states.
    self.lastState = nil
    self.lastAction = nil
    self.v_avg = 0 -- V running average.
    self.tderr_avg = 0 -- TD error running average.

    self.q_max = 1
    self.r_max = 1

    ----------------------------------------------------------------------------

    -- get modules
    self.enc = self.network.modules[1]
    self.dec = self.network.modules[2]

    self.p_enc = self.pred_net.modules[1]
    self.p_dec = self.pred_net.modules[2]

    self.enc_w, self.enc_dw = self.enc:getParameters()
    self.dec_w, self.dec_dw = self.dec:getParameters()
    self.p_dec_w, self.p_dec_dw = self.p_dec:getParameters()

    -- share encoder params, not decoder params
    -- p_enc has a ParallelTable
    self.p_enc.modules[1]:share(self.enc,
                'weight', 'bias', 'gradWeight', 'gradBias')
    self.p_enc.modules[2]:share(self.enc,
                'weight', 'bias', 'gradWeight', 'gradBias')

    -- -- encoder (shared)
    self.enc_dw:zero()
    self.enc_deltas = self.enc_dw:clone():fill(0)
    self.enc_tmp= self.enc_dw:clone():fill(0)
    self.enc_g  = self.enc_dw:clone():fill(0)
    self.enc_g2 = self.enc_dw:clone():fill(0)

    -- linear dqn
    self.dec_dw:zero()
    self.dec_deltas = self.dec_dw:clone():fill(0)
    self.dec_tmp= self.dec_dw:clone():fill(0)
    self.dec_g  = self.dec_dw:clone():fill(0)
    self.dec_g2 = self.dec_dw:clone():fill(0)

    -- autoencoder decoder
    self.p_dec_dw:zero()

    ----------------------------------------------------------------------------

    -- the target_network is for doing q learning updates. This is because
    -- we are not doing tabular updates. So we need an "old" network to
    -- compare our updates with

    if self.target_q then
        self.target_network = self.network:clone()
    end
end


function nql:reset(state)
    if not state then
        return
    end
    self.best_network = state.best_network
    self.network = state.model

    ----------------------------------------------------------------------------
    self.enc_w, self.enc_dw = self.enc:getParameters()
    self.dec_w, self.dec_dw = self.dec:getParameters()
    self.p_dec_w, self.p_dec_dw = self.p_dec:getParameters()

    self.p_enc.modules[1]:share(self.enc,
                'weight', 'bias', 'gradWeight', 'gradBias')
    self.p_enc.modules[2]:share(self.enc,
                'weight', 'bias', 'gradWeight', 'gradBias')

    self.enc_dw:zero()  -- self.network
    self.dec_dw:zero()  -- self.network
    self.p_dec_dw:zero()  -- self.pred_net

    ----------------------------------------------------------------------------
    self.numSteps = 0
    print("RESET STATE SUCCESFULLY")
end


function nql:preprocess(rawstate)
    if self.preproc then
        return self.preproc:forward(rawstate:float())
                    :clone():reshape(self.state_dim)
    end

    return rawstate
end


function nql:getQUpdate(args)
    local s, a, r, s2, term, delta
    local q, q2, q2_max

    s = args.s
    a = args.a
    r = args.r
    s2 = args.s2
    term = args.term

    if udcign_reshape then
        s = s:resize(s:nElement()/84/84,1,84,84)
        s2 = s2:resize(s2:nElement()/84/84,1,84,84)
    end

    -- The order of calls to forward is a bit odd in order
    -- to avoid unnecessary calls (we only need 2).

    -- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
    term = term:clone():float():mul(-1):add(1)

    local target_q_net
    if self.target_q then
        target_q_net = self.target_network
    else
        target_q_net = self.network
    end

    target_q_net:clearState()
    collectgarbage()
    collectgarbage()

    -- Compute max_a Q(s_2, a).
    -- q2_max = target_q_net:forward(s2):float():max(2)  -- getting an error here
    if udcign_reshape then
        local bsize = s2:nElement()/84/84/4
        local encout = target_q_net.modules[1]:forward(s2)
        encout = encout:resize(bsize,800) -- hardcoded
        q2_max = target_q_net.modules[2]:forward(encout):float():max(2)
    else
        q2_max = target_q_net:forward(s2):float():max(2)  -- getting an error here
    end

    -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
    q2 = q2_max:clone():mul(self.discount):cmul(term)

    delta = r:clone():float()

    if self.rescale_r then
        delta:div(self.r_max)
    end
    delta:add(q2)

    self.network:clearState()
    collectgarbage()
    collectgarbage()

    -- q = Q(s,a)
    -- local q_all = self.network:forward(s):float()
    local q_all
    if udcign_reshape then
        local bsize = s:nElement()/84/84/4
        local encout = self.network.modules[1]:forward(s)
        encout = encout:resize(bsize,800) -- hardcoded
        q_all = self.network.modules[2]:forward(encout):float()
    else
        q_all = self.network:forward(s2):float()  -- getting an error here
    end

    q = torch.FloatTensor(q_all:size(1))
    for i=1,q_all:size(1) do
        q[i] = q_all[i][a[i]]
    end
    delta:add(-1, q)

    if self.clip_delta then
        delta[delta:ge(self.clip_delta)] = self.clip_delta
        delta[delta:le(-self.clip_delta)] = -self.clip_delta
    end

    -- targets is a table
    local targets = torch.zeros(self.minibatch_size, self.n_actions):float()
    for i=1,math.min(self.minibatch_size,a:size(1)) do
        targets[i][a[i]] = delta[i]
    end

    if self.gpu >= 0 then targets = targets:cuda() end

    return targets, delta, q2_max
end


function nql:qLearnMinibatch()
    -- print('learn minibatch')
    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
    assert(self.transitions:size() > self.minibatch_size)

    local s, a, r, s2, term = self.transitions:sample(self.minibatch_size)

    local targets, delta, q2_max = self:getQUpdate{s=s, a=a, r=r, s2=s2,
        term=term, update_qmax=true}

    -- -- -- zero gradients of parameters
    -- self.enc_dw:zero()
    -- self.dec_dw:zero()
    -- self.p_dec_dw:zero()
    --
    -- -- mutate the scheduler_iteration
    -- self.predictive_iteration = self.predictive_iteration+1
    -- self.p_args.p_scheduler_iteration[1] = self.p_args.p_scheduler_iteration[1]+1
    --
    -- -- first split into batches
    -- local s_reshaped = s:reshape(self.minibatch_size, self.hist_len, 84, 84)
    -- local s_pairs = {}
    -- table.insert(s_pairs, {s_reshaped[{{},{1}}],s_reshaped[{{},{2}}]})
    -- table.insert(s_pairs, {s_reshaped[{{},{2}}],s_reshaped[{{},{3}}]})
    -- table.insert(s_pairs, {s_reshaped[{{},{3}}],s_reshaped[{{},{4}}]})
    --
    -- for k,s_pair in pairs(s_pairs) do
    --
    --     -- zero grad params before rmsprop
    --     local loss, _ = self:feval(s_pair)
    --
    --     function feval_encoder(x,input)
    --         return loss, self.enc_dw
    --     end
    --     function feval_decoder(x,input)
    --         return loss, self.p_dec_dw
    --     end
    --
    --     rmsprop(feval_encoder, s_pair, self.enc_w, self.enc_optim_state)
    --     rmsprop(feval_decoder, s_pair, self.p_dec_w, self.dec_optim_state)
    --
    --     -- print self.enc_dw here
    --     dw_ratio[1] = dw_ratio[1]+self.enc_dw:norm()
    -- end
    --
    -- -- here we do updates on learning_rate if needed
    -- if self.predictive_iteration % self.p_learning_rate_decay_interval == 0
    --                                     and self.p_learning_rate_decay < 1 then
    --     if self.predictive_iteration >= self.p_learning_rate_decay_after then
    --         self.optim_state.learningRate = self.optim_state.learningRate
    --                                                 * self.p_learning_rate_decay
    --         print('decayed function learning rate by a factor ' ..
    --                         self.p_learning_rate_decay .. ' to '
    --                         .. self.optim_state.learningRate)
    --     end
    -- end

    ----------------------------------------------------------------------------
    -- get new gradient

    if udcign_reshape then
        s = s:resize(s:nElement()/84/84,1,84,84)
    end

    -- zero gradients of parameters before updating dqn
    self.enc_dw:zero()
    self.dec_dw:zero()

    if fix_pre_encoder then
        if udcign_reshape then
            local bsize = s:nElement()/84/84/4
            local encout = self.network.modules[1].output  -- necessary to clone()?
            encout = encout:resize(bsize,800) -- hardcoded  resize or reshape?
            self.network.modules[2]:backward(encout,targets)
        else
            self.network.modules[2]:backward(self.network.modules[1].output,
                                        targets)
        end
    else
        if udcign_reshape then
            local bsize = s:nElement()/84/84/4
            local encout = self.network.modules[1].output  -- necessary to clone()?
            encout = encout:resize(bsize,800) -- hardcoded  resize or reshape?
            local grad_encout = self.network.modules[2]:backward(encout,targets)
            grad_encout = grad_encout:resize(bsize*4,200)
            self.network.modules[1]:backward(s,grad_encout)
        else
            self.network:backward(s, targets)
        end
    end

    self.enc_dw:mul(self.p_lambda)

    dw_ratio[2] = dw_ratio[2]+self.enc_dw:norm()


    -- add weight cost to gradient
    self.enc_dw:add(-self.wc, self.enc_w)
    self.dec_dw:add(-self.wc, self.dec_w)

    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
                self.lr_end
    self.lr = math.max(self.lr, self.lr_end)

    -- use gradients
    self.enc_g:mul(0.95):add(0.05, self.enc_dw)
    self.enc_tmp:cmul(self.enc_dw, self.enc_dw)
    self.enc_g2:mul(0.95):add(0.05, self.enc_tmp)
    self.enc_tmp:cmul(self.enc_g, self.enc_g)
    self.enc_tmp:mul(-1)
    self.enc_tmp:add(self.enc_g2)
    self.enc_tmp:add(0.01)
    self.enc_tmp:sqrt()

    self.dec_g:mul(0.95):add(0.05, self.dec_dw)
    self.dec_tmp:cmul(self.dec_dw, self.dec_dw)
    self.dec_g2:mul(0.95):add(0.05, self.dec_tmp)
    self.dec_tmp:cmul(self.dec_g, self.dec_g)
    self.dec_tmp:mul(-1)
    self.dec_tmp:add(self.dec_g2)
    self.dec_tmp:add(0.01)
    self.dec_tmp:sqrt()

    -- accumulate update
    self.enc_deltas:mul(0):addcdiv(self.lr, self.enc_dw, self.enc_tmp)
    self.enc_w:add(self.enc_deltas)

    self.dec_deltas:mul(0):addcdiv(self.lr, self.dec_dw, self.dec_tmp)
    self.dec_w:add(self.dec_deltas)

    if self.gpu and self.gpu >= 0 then
        cutorch.synchronize()
    end
    collectgarbage()
end

function nql:qLearnEnvironment()
    -- print('learn env')
    -- assert(false,'Did you divide ratio by pred_learn_freq and initialize pred_learn_freq? and call qLearnEnvironment every pred_learn_freq times? pred_learn_freq shoudl be in train_predictive_agent')
    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
    -- self.pred_net:clearState()
    -- if self.gpu and self.gpu >= 0 then
    --     cutorch.synchronize()
    -- end
    -- collectgarbage()
    -- self.pred_net:clearState()
    -- if self.gpu and self.gpu >= 0 then
    --     cutorch.synchronize()
    -- end
    -- collectgarbage()

    local pred_batch_size = self.minibatch_size*global_args.learn_freq
    assert(self.transitions:size() > pred_batch_size)
    local s, a, r, s2, term = self.transitions:sample(pred_batch_size)
    a = nil
    r = nil
    s2 = nil
    collectgarbage()
    -- self.pred_net:clearState()
    -- if self.gpu and self.gpu >= 0 then
    --     cutorch.synchronize()
    -- end
    -- collectgarbage()

    -- zero gradients of parameters
    self.enc_dw:zero()
    self.dec_dw:zero()
    self.p_dec_dw:zero()

    -- mutate the scheduler_iteration
    self.predictive_iteration = self.predictive_iteration+1
    self.p_args.p_scheduler_iteration[1] = self.p_args.p_scheduler_iteration[1]+1

    -- first split into batches
    local s_reshaped = s:resize(pred_batch_size, self.hist_len, 84, 84)
    local s_pairs = {}
    -- table.insert(s_pairs, {s_reshaped[{{},{1}}],s_reshaped[{{},{2}}]})
    -- table.insert(s_pairs, {s_reshaped[{{},{2}}],s_reshaped[{{},{3}}]})
    table.insert(s_pairs, {s_reshaped[{{},{3}}],s_reshaped[{{},{4}}]})

    -- s_pair = s:resize(s:nElement()/84/84,1,84,84)


    for k,s_pair in pairs(s_pairs) do

        -- zero grad params before rmsprop
        local loss, _ = self:feval(s_pair)

        function feval_encoder(x,input)
            return loss, self.enc_dw
        end
        function feval_decoder(x,input)
            return loss, self.p_dec_dw
        end

        rmsprop(feval_encoder, s_pair, self.enc_w, self.enc_optim_state)
        rmsprop(feval_decoder, s_pair, self.p_dec_w, self.dec_optim_state)

        -- print self.enc_dw here
        dw_ratio[1] = dw_ratio[1]+self.enc_dw:norm()
    end

    -- here we do updates on learning_rate if needed
    if self.predictive_iteration % self.p_learning_rate_decay_interval == 0
                                        and self.p_learning_rate_decay < 1 then
        if self.predictive_iteration >= self.p_learning_rate_decay_after then
            self.enc_optim_state.learningRate = self.enc_optim_state.learningRate
                                                    * self.p_learning_rate_decay
            self.dec_optim_state.learningRate = self.dec_optim_state.learningRate
                                                    * self.p_learning_rate_decay
            print('decayed function learning rate by a factor ' ..
                            self.p_learning_rate_decay .. ' to '
                            .. self.enc_optim_state.learningRate)
        end
    end

    if self.gpu and self.gpu >= 0 then
        cutorch.synchronize()
    end
    collectgarbage()
end

-- feval for full_udcign
-- do fwd/bwd and return loss, grad_params
function nql:feval(input)

    ------------------- forward pass -------------------
    self.pred_net:training() -- make sure we are in correct mode

    self.enc_dw:zero()
    self.dec_dw:zero()
    self.p_dec_dw:zero()

    local output = self.pred_net:forward(input)
    local loss = self.pred_criterion:forward(output, input[2])
    local grad_output = self.pred_criterion:backward(output, input[2]):clone()
    self.pred_net:backward(input,grad_output)

    ------------------ regularize -------------------
    if self.p_L2 > 0 then
        print('regularize')
        -- Loss:
        loss = loss + self.p_L2 *
                            torch.cat{self.enc_w, self.p_dec_w}:norm(2)^2/2
        -- Gradients:
        -- grad_params:add( params:clone():mul(opt.L2) )

        self.enc_dw:add( self.enc_w:clone():mul(opt.L2) )
        self.p_dec_dw:add( self.p_dec_w:clone():mul(opt.L2) )
    end

    self.enc_dw:clamp(-self.p_grad_clip, self.p_grad_clip)
    self.p_dec_dw:clamp(-self.p_grad_clip, self.p_grad_clip)

    collectgarbage()
    return loss, torch.cat{self.enc_dw,self.p_dec_dw}  -- not actually used
end


function nql:sample_validation_data()
    local s, a, r, s2, term = self.transitions:sample(self.valid_size)

    -- michael added this
    if self.gpu and self.gpu >= 0 then
        cutorch.synchronize()
    end
    collectgarbage()

    -- so, is it necessary to clone these things?
    self.valid_s    = s:clone()
    self.valid_a    = a:clone()
    self.valid_r    = r:clone()
    self.valid_s2   = s2:clone() -- cuda runtime error because of memory
    self.valid_term = term:clone()
end


function nql:compute_validation_statistics(split)
    if not split then
        local targets, delta, q2_max = self:getQUpdate{s=self.valid_s,
            a=self.valid_a, r=self.valid_r, s2=self.valid_s2, term=self.valid_term}
        self.v_avg = self.q_max * q2_max:mean()
        self.tderr_avg = delta:clone():abs():mean() -- note that :abs() mutates!
    else
        print('split')
        -- do a for loop to do this iteratively
        assert(self.valid_size % 10 == 0)
        local sub_valid_size = self.valid_size/10
        local q2_max_sum = 0 -- change this to a torch tensor of 0s  the same size as delta
        local delta_sum = 0  -- change this to a torch tensor of 0s  the same size as delta

        -- delta = (valid_size x 1)
        -- q2_max_sum = (valid_size)
        --
        for i = 1,self.valid_size,sub_valid_size do
            local _ , delta, q2_max = self:getQUpdate{
                        s=self.valid_s[{{i,i+sub_valid_size-1}}],
                        a=self.valid_a[{{i,i+sub_valid_size-1}}],
                        r=self.valid_r[{{i,i+sub_valid_size-1}}],
                        s2=self.valid_s2[{{i,i+sub_valid_size-1}}],
                        term=self.valid_term[{{i,i+sub_valid_size-1}}]}
            q2_max_sum = q2_max_sum + q2_max:sum()
            delta_sum = delta_sum + delta:clone():abs():sum()
            collectgarbage()
        end

        self.v_avg = self.q_max * q2_max_sum/self.valid_size
        self.tderr_avg = delta_sum/self.valid_size
    end
end


function nql:perceive(reward, rawstate, terminal, testing, testing_ep)
    -- Preprocess state (will be set to nil if terminal)
    local state = self:preprocess(rawstate):float()  -- TODO: THIS IS PREPROCESS

    local curState

    if self.max_reward then
        reward = math.min(reward, self.max_reward)
    end
    if self.min_reward then
        reward = math.max(reward, self.min_reward)
    end
    if self.rescale_r then
        self.r_max = math.max(self.r_max, reward)
    end

    self.transitions:add_recent_state(state, terminal)

    local currentFullState = self.transitions:get_recent()

    --Store transition s, a, r, s'
    if self.lastState and not testing then
        self.transitions:add(self.lastState, self.lastAction, reward,
                             self.lastTerminal, priority)
    end

    if self.numSteps == self.learn_start+1 and not testing then
         -- we do stuff on validation data instead here
        self:sample_validation_data()
    end

    curState= self.transitions:get_recent()
    curState = curState:resize(1, unpack(self.input_dims))  -- fix this! You have to change input_dims

    -- Select action
    local actionIndex = 1
    if not terminal then
        actionIndex = self:eGreedy(curState, testing_ep)
    end

    self.transitions:add_recent_action(actionIndex)

    --Do some Q-learning updates
    if self.numSteps > self.learn_start and not testing and
        self.numSteps % self.update_freq == 0 then
        for i = 1, self.n_replay do
            self:qLearnMinibatch() -- do this once each 4 perceives
        end
    end

    if self.numSteps > self.learn_start and not testing and
        self.numSteps % (self.update_freq*global_args.learn_freq) == 0 then
        self:qLearnEnvironment() -- do this once each time we perceive
    end

    if not testing then
        self.numSteps = self.numSteps + 1
    end

    self.lastState = state:clone()
    self.lastAction = actionIndex
    self.lastTerminal = terminal

    if self.target_q and self.numSteps % self.target_q == 1 then
        self.network:clearState()
        self.target_network = self.network:clone()
    end

    if self.gpu and self.gpu >= 0 then
        cutorch.synchronize()
    end
    collectgarbage()

    if not terminal then
        return actionIndex
    else
        return 0
    end
end


function nql:eGreedy(state, testing_ep)
    self.ep = testing_ep or (self.ep_end +
                math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                math.max(0, self.numSteps - self.learn_start))/self.ep_endt))
    -- Epsilon greedy
    if torch.uniform() < self.ep then
        return torch.random(1, self.n_actions)
    else
        return self:greedy(state)
    end
end


function nql:greedy(state)
    -- Turn single state into minibatch.  Needed for convolutional nets.
    if state:dim() == 2 then
        assert(false, 'Input must be at least 3D')
        state = state:resize(1, state:size(1), state:size(2))
    end

    if self.gpu >= 0 then
        state = state:cuda()
    end

    -- local q = self.network:forward(state):float():squeeze()
    local q
    if udcign_reshape then
        state = state:resize(state:nElement()/84/84,1,84,84) -- hardcoded
        local bsize = state:nElement()/84/84/4
        local encout = self.network.modules[1]:forward(state)
        encout = encout:resize(bsize,800) -- hardcoded
        q = self.network.modules[2]:forward(encout):float():squeeze()
    else
        q = self.network:forward(state):float():squeeze()
    end

    local maxq = q[1]
    local besta = {1}

    -- Evaluate all other actions (with random tie-breaking)
    for a = 2, self.n_actions do
        if q[a] > maxq then
            besta = { a }
            maxq = q[a]
        elseif q[a] == maxq then
            besta[#besta+1] = a
        end
    end
    self.bestq = maxq

    local r = torch.random(1, #besta)

    self.lastAction = besta[r]

    return besta[r]
end


function nql:createNetwork()
    local n_hid = 128
    local mlp = nn.Sequential()
    mlp:add(nn.Reshape(self.hist_len*self.ncols*self.state_dim))
    mlp:add(nn.Linear(self.hist_len*self.ncols*self.state_dim, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, self.n_actions))

    return mlp
end


function nql:_loadNet()
    local net = self.network
    if self.gpu then
        net:cuda()
    else
        net:float()
    end
    return net
end


function nql:init(arg)
    self.actions = arg.actions
    self.n_actions = #self.actions
    self.network = self:_loadNet()
    -- Generate targets.
    self.transitions:empty()
end


function nql:report()
    if self.predictive_iteration > 0 then
        print('pred/dqn:', dw_ratio[1]*global_args.learn_freq/dw_ratio[2]/3)
    end
    print(get_weight_norms(self.network))
    print(get_grad_norms(self.network))
end
