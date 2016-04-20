--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

local optnet = require 'optnet'
require 'rmsprop'

if not dqn then
    require 'initenv'
end

local nql = torch.class('dqn.NeuralQPredictiveLearner')

-- false for backpropagating all the way
-- true for backpropagating only through the linear
local fix_pre_encoder = true

function nql:__init(args)
    self.state_dim  = args.state_dim -- State dimensionality.
    self.actions    = args.actions
    self.n_actions  = #self.actions
    self.verbose    = args.verbose
    self.best       = args.best

    --- epsilon annealing
    self.ep_start   = args.ep or 1
    self.ep         = self.ep_start -- Exploration probability.
    self.ep_end     = args.ep_end or self.ep
    self.ep_endt    = args.ep_endt or 1000000

    ---- learning rate annealing
    self.lr_start       = args.lr or 0.01 --Learning rate.
    self.lr             = self.lr_start
    self.lr_end         = args.lr_end or self.lr
    self.lr_endt        = args.lr_endt or 1000000
    self.wc             = args.wc or 0  -- L2 weight cost.
    self.minibatch_size = args.minibatch_size or 1
    self.valid_size     = args.valid_size or 500

    --- Q-learning parameters
    self.discount       = args.discount or 0.99 --Discount factor.
    self.update_freq    = args.update_freq or 1
    -- Number of points to replay per learning step.
    self.n_replay       = args.n_replay or 1
    -- Number of steps after which learning starts.
    self.learn_start    = args.learn_start or 0
     -- Size of the transition table.
    self.replay_memory  = args.replay_memory or 1000000
    self.hist_len       = args.hist_len or 1
    self.rescale_r      = args.rescale_r
    self.max_reward     = args.max_reward
    self.min_reward     = args.min_reward
    self.clip_delta     = args.clip_delta
    self.target_q       = args.target_q
    self.bestq          = 0

    self.gpu            = args.gpu

    self.ncols          = args.ncols or 1  -- number of color channels in input
    self.input_dims     = args.input_dims or {self.hist_len*self.ncols, 84, 84}  -- this incorporates hist_len!
    -- self.input_dims     = args.input_dims or {self.hist_len,3, 210, 160}  -- this incorporates hist_len!  -- TODO
    self.preproc        = args.preproc  -- name of preprocessing network
    self.histType       = args.histType or "linear"  -- history type to use
    self.histSpacing    = args.histSpacing or 1
    self.nonTermProb    = args.nonTermProb or 1
    self.bufferSize     = args.bufferSize or 512

    self.transition_params = args.transition_params or {}

    self.network    = args.network or self:createNetwork() -- args.network is loaded by run_gpu as the convnet_atari3
    -- self.fix_pre_encoder = false

    -- check whether there is a network file
    local network_function
    if not (type(self.network) == 'string') then
        error("The type of the network provided in NeuralQLearner" ..
              " is not a string!")
    end

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

    -- here, iniitialize predictive network
    self.pred_net = args.pred_net
    self.pred_criterion = nn.MotionBCECriterion(args.motion_scale)

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
        self.pred_net:cuda()
    else
        self.network:float()
        self.pred_net:float()
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
        self.tensor_type = torch.CudaTensor
    else
        self.network:float()
        self.pred_net:float()
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
    -- parameters for self.network

    -- self.w, self.dw = self.network:getParameters()
    -- self.dw:zero()
    --
    -- self.deltas = self.dw:clone():fill(0)
    --
    -- self.tmp= self.dw:clone():fill(0)
    -- self.g  = self.dw:clone():fill(0)
    -- self.g2 = self.dw:clone():fill(0)

    self.dec = self.network.modules[2]
    self.p_dec = self.prd_net.modules[2]

    -- encoder
    self.enc_w, self.enc_dw = self.network.modules[1]:getParameters()
    self.enc_dw:zero()

    self.enc_deltas = self.enc_dw:clone():fill(0)

    self.enc_tmp= self.enc_dw:clone():fill(0)
    self.enc_g  = self.enc_dw:clone():fill(0)
    self.enc_g2 = self.enc_dw:clone():fill(0)

    -- linear
    self.dec_w, self.dec_dw = self.network.modules[2]:getParameters()
    self.dec_dw:zero()

    self.dec_deltas = self.dec_dw:clone():fill(0)

    self.dec_tmp= self.dec_dw:clone():fill(0)
    self.dec_g  = self.dec_dw:clone():fill(0)
    self.dec_g2 = self.dec_dw:clone():fill(0)

    ----------------------------------------------------------------------------


    -- the target_network is for doing q learning updates. This is because
    -- we are not doing tabular updates. So we need an "old" network to
    -- compare our updates with

    if self.target_q then
        self.target_network = self.network:clone()
    end


    ----------------------------------------------------------------------------
    -- parameters for self.pred_net

    -- here do other initialization for the target q network.
    self.pred_net.modules[1]:share(self.network.modules[1],'weight', 'bias'))
    self.p_dec_w, self.p_dec_dw = self.pred_net.modules[2].getParameters()
    ----------------------------------------------------------------------------
end


function nql:reset(state)
    if not state then
        return
    end
    self.best_network = state.best_network
    self.network = state.model

    ----------------------------------------------------------------------------
    -- parameters for self.network
    self.enc_w, self.enc_dw = self.network.modules[1]:getParameters()
    self.enc_dw:zero()

    self.dec_w, self.dec_dw = self.network.modules[2]:getParameters()
    self.dec_dw:zero()

    ----------------------------------------------------------------------------
    -- parameters for self.pred_net
    self.pred_net.modules[1]:share(self.network.modules[1],'weight', 'bias'))
    self.p_dec_w, self.p_dec_dw = self.pred_net.modules[2].getParameters()

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
    q2_max = target_q_net:forward(s2):float():max(2)  -- getting an error here


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
    local q_all = self.network:forward(s):float()
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
    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
    assert(self.transitions:size() > self.minibatch_size)

    local s, a, r, s2, term = self.transitions:sample(self.minibatch_size)

    local targets, delta, q2_max = self:getQUpdate{s=s, a=a, r=r, s2=s2,
        term=term, update_qmax=true}

    -- zero gradients of parameters
    self.enc_dw:zero()
    self.dec_dw:zero()

    -- get new gradient



    --
    -- here do forward on the ICLR, as well as backward using a closure (rmsprop)
    -- you can just modify rmsprop to take in inptu as argument
    -- define feval outside: feval takes in _params, as well as input
    -- in feval we should scale gradients
    -- do a call to rmsprop, which does a forward and backward pass
    -- print gradOutput of the linears for the dqn encoder as well as iclr encoder
    -- to compare grad_norms.

    -- optim_state
    -- this does a forward and backward on the pred_net.
    -- this will automatically update the self.enc_w and self.p_dec_w.
    -- note that the grad params for the self.pred_net are completely different
    -- from that of self.network
    -- after this, we will update self.enc_w again, as well as self.dec_w
    local new_params, _ = rmsprop(feval, s, torch.cat{self.enc_w,self.p_dec_w},
                                                                    optim_state)
    self.enc_w:copy(new_params[{{1,self.enc_w:size(1)}}])
    self.p_dec_w:copy(new_params[{{self.enc_w:size(1)+1,-1}}])


    if fix_pre_encoder then
        self.network.modules[2]:backward(self.network.modules[1].output,
                                        targets)
    else
        self.network:backward(s, targets)
    end

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


    -- here, print out the gradInput of the decs
    print('Find Lambda')
    print(self.dec.gradInput:norm())
    print(self.p_dec.gradInput:norm())

    if self.gpu and self.gpu >= 0 then
        cutorch.synchronize()
    end
    collectgarbage()
end

-- feval for full_udcign
-- do fwd/bwd and return loss, grad_params
-- TODO: should input already be reshaped, or no?
-- expect the input to be
function feval(x, input, lambda)
    assert(x:size(1) == self.enc_w:size(1) + self.p_dec_w:size(1))
    if x[{{1,self.enc_w:size(1)}}] ~= self.enc_w then
        error("Params not equal to given feval argument.")
        self.enc_w:copy(x[{{1,self.enc_w:size(1)}}])
    end
    if x[{{self.enc_w:size(1)+1,-1}}] ~= self.p_dec_w then
        error("Params not equal to given feval argument.")
        self.p_dec_w:copy(x[{{self.enc_w:size(1)+1,-1}}])
    end
    -- how do I extract the params of the pred_net only?
    -- TODO: I'm going to have to do something with the sizes of the params

    -- grad_params:zero()
    -- TODO: maybe I don't need to do grad_params, since I bascially do
    -- two backwards because gradients accumulate

    ------------------ get minibatch -------------------
    -- local input = data_loaders.load_random_atari_batch('train')

    ------------------- forward pass -------------------
    self.pred_net:training() -- make sure we are in correct mode

    local loss
    local output = self.pred_net:forward(input)

    loss = self.pred_criterion:forward(output, input[2])
    local grad_output = self.pred_criterion:backward(output, input[2]):clone()

    self.pred_net:backward(input, grad_output)


    ------------------ regularize -------------------
    if opt.L2 > 0 then
        -- Loss:
        loss = loss + opt.coefL2 *
                            torch.cat{self.enc_w, self.p_dec_w}:norm(2)^2/2
        -- Gradients:
        -- grad_params:add( params:clone():mul(opt.L2) )

        self.enc_dw:add( self.enc_w:clone():mul(opt.L2) )
        self.p_dec_dw:add( self.p_dec_w:clone():mul(opt.L2) )
    end

    -- TODO: not sure if this clamp works
    self.enc_dw::clamp(-opt.grad_clip, opt.grad_clip)
    self.p_dec_dw:clamp(-opt.grad_clip, opt.grad_clip)

    collectgarbage()
    return loss, grad_params
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
            self:qLearnMinibatch() -- do this once each time we perceive
        end
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

    local q = self.network:forward(state):float():squeeze()
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
    print(get_weight_norms(self.network))
    print(get_grad_norms(self.network))
end
