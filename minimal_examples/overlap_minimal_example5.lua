-- sharing weights + fwd/bwd + parallel + in functions + in class + rmsprop

require 'nn'

local minimal = {}
minimal.__index = minimal

function minimal:create(args)
    local self = {}
    setmetatable(self, minimal)

    -- Create networks
    local net1 = nn.Sequential()
    net1:add(nn.Linear(2,2))
    net1:add(nn.Linear(2,1))

    local net2 = nn.Sequential()
    local net2parallel = nn.ParallelTable()
    net2parallel:add(nn.Linear(2,2))
    net2parallel:add(nn.Linear(2,2))
    net2:add(net2parallel)
    net2:add(nn.CAddTable())
    net2:add(nn.Linear(2,2))

    self.crit = nn.MSECriterion()

    self.net1 = net1
    self.net2 = net2

    print('net1')
    print(self.net1)
    print('net2')
    print(self.net2)

    -- Get modules
    self.enc1 = net1.modules[1]
    self.dec1 = net1.modules[2]
    self.enc2 = net2.modules[1]
    self.dec2 = net2.modules[3]


    -- Get Parameters
    self.enc_w, self.enc_dw = self.enc1:getParameters()
    self.dec1_w, self.dec1_dw = self.dec1:getParameters()
    self.dec2_w, self.dec2_dw = self.dec2:getParameters()

    -- Share NOTE You may have to do a ParallelTable here instead
    self.enc2.modules[1]:share(self.enc1,
                                        'weight','bias','gradWeight','gradBias')
    self.enc2.modules[2]:share(self.enc1,
                                        'weight','bias','gradWeight','gradBias')

    -- Zero Grad Parameters
    self.enc_dw:zero()
    self.dec1_dw:zero()
    self.dec2_dw:zero()

    -- Intermediate containers
    self.enc_deltas = self.enc_dw:clone():fill(0)
    self.enc_tmp= self.enc_dw:clone():fill(0)
    self.enc_g  = self.enc_dw:clone():fill(0)
    self.enc_g2 = self.enc_dw:clone():fill(0)

    -- Clone net1
    self.net1clone = self.net1:clone()

    return self
end


-- Functions
function minimal:net1forward()
    local input1 = torch.rand(3,2)

    -- Clone net1
    local out1 = self.net1:forward(input1)
end

function minimal:feval(params,funcinput2)
    -- Zero Grad Parameters
    self.enc_dw:zero()
    self.dec1_dw:zero()
    self.dec2_dw:zero()


    -- Inspect Grad Parameters
    print('Before fwd/bwd: These should be equal')
    print(self.enc_dw:sum()..'\tenc_dw:sum()')
    print(self.enc2.modules[1].gradWeight:sum()+
            self.enc2.modules[1].gradBias:sum()..
            '\tnet2.modules[1].modules[1].gradWeight:sum()')
    print(self.enc2.modules[2].gradWeight:sum()+
            self.enc2.modules[2].gradBias:sum()..
            '\tnet2.modules[1].modules[2].gradWeight:sum()')


    -- Forward on net2
    local out2 = self.net2:forward(funcinput2)
    local loss = self.crit:forward(out2,funcinput2[2])


    -- Backward on net2
    local gout2 = self.crit:backward(out2, funcinput2[2])
    local gin2 = self.net2:backward(funcinput2,gout2)


    print('After fwd/bwd: These should be equal')
    print(self.enc_dw:sum()..'\tenc_dw:sum()')
    print(self.enc2.modules[1].gradWeight:sum()+
            self.enc2.modules[1].gradBias:sum()..
            '\tnet2.modules[1].modules[1].gradWeight:sum()')
    print(self.enc2.modules[2].gradWeight:sum()+
            self.enc2.modules[2].gradBias:sum()..
            '\tnet2.modules[1].modules[2].gradWeight:sum()')
    return loss, torch.cat({self.enc_dw,self.dec2_dw})
end

function minimal:getQUpdate()
    -- net1 forward
    for t = 1,3 do
        self:net1forward()
    end
end

function minimal:qLearnMinibatch(params)
    -- Zero Grad Parameters
    self.enc_dw:zero()
    self.dec1_dw:zero()
    self.dec2_dw:zero()

    local optim_state = {learningRate=0.1, alpha=0.97}

    -- net2 fwd bwd
    for t = 1,3 do
        print(t)
        local input2 = {torch.rand(3,2),torch.rand(3,2)}
        local newp, _ = self:rmsprop(_,input2,torch.cat{self.enc_w,self.dec2_w},optim_state)

        self.enc_w:copy(newp[{{1,self.enc_w:size(1)}}])
        self.dec2_w:copy(newp[{{self.enc_w:size(1)+1,-1}}])

        print('After updating weights. These should be equal')
        print(self.enc_w:sum()..'\tself.enc_w:sum()')
        print(self.enc2.modules[1].weight:sum()+
                self.enc2.modules[1].bias:sum()..
                '\tnet2.modules[1].modules[1].weight:sum()')
        print(self.enc2.modules[2].weight:sum()+
                self.enc2.modules[2].bias:sum()..
                '\tnet2.modules[1].modules[2].weight:sum()')
    end
end

function minimal:rmsprop(opfunc, input, x, config, state)
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-2
    local alpha = config.alpha or 0.99
    local epsilon = config.epsilon or 1e-8
    local wd = config.weightDecay or 0

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = self:feval(x, input)  -- hardcoded

    -- (2) weight decay
    if wd ~= 0 then
      dfdx:add(wd, x)
    end

    -- (3) initialize mean square values and square gradient storage
    if not state.m then
      state.m = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
      state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
    end

    -- (4) calculate new (leaky) mean squared values
    state.m:mul(alpha)
    state.m:addcmul(1.0-alpha, dfdx, dfdx)

    -- (5) perform update
    state.tmp:sqrt(state.m):add(epsilon)
    x:addcdiv(-lr, dfdx, state.tmp)

    -- return x*, f(x) before optimization
    return x, {fx}
end


local m = minimal.create()
m:getQUpdate()
m:qLearnMinibatch()
