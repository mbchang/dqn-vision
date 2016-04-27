-- sharing weights + fwd/bwd + parallel + in functions + in class (rmsprop next)

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
    self.enc = net1.modules[1]
    self.dec1 = net1.modules[2]
    self.dec2 = net2.modules[3]


    -- Get Parameters
    self.enc_w, self.enc_dw = self.enc:getParameters()
    self.dec1_w, self.dec1_dw = self.dec1:getParameters()
    self.dec2_w, self.dec2_dw = self.dec2:getParameters()

    -- Share NOTE You may have to do a ParallelTable here instead
    self.net2.modules[1].modules[1]:share(self.enc,
                                        'weight','bias','gradWeight','gradBias')
    self.net2.modules[1].modules[2]:share(self.enc,
                                        'weight','bias','gradWeight','gradBias')

    -- Zero Grad Parameters
    self.enc_dw:zero()
    self.dec1_dw:zero()
    self.dec2_dw:zero()

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

function minimal:net2forwardbackward(funcinput2)
    -- Zero Grad Parameters
    self.enc_dw:zero()
    self.dec1_dw:zero()
    self.dec2_dw:zero()


    -- Inspect Grad Parameters
    print('Before fwd/bwd: These should be equal')
    print(self.enc_dw:sum()..'\tenc_dw:sum()')
    print(self.net2.modules[1].modules[1].gradWeight:sum()+
            self.net2.modules[1].modules[1].gradBias:sum()..
            '\tnet2.modules[1].modules[1].gradWeight:sum()')
    print(self.net2.modules[1].modules[2].gradWeight:sum()+
            self.net2.modules[1].modules[2].gradBias:sum()..
            '\tnet2.modules[1].modules[2].gradWeight:sum()')


    -- Forward on net2
    local out2 = self.net2:forward(funcinput2)
    local loss = self.crit:forward(out2,funcinput2[2])


    -- Backward on net2
    local gout2 = self.crit:backward(out2, funcinput2[2])
    local gin2 = self.net2:backward(funcinput2,gout2)


    print('After fwd/bwd: These should be equal')
    print(self.enc_dw:sum()..'\tenc_dw:sum()')
    print(self.net2.modules[1].modules[1].gradWeight:sum()+
            self.net2.modules[1].modules[1].gradBias:sum()..
            '\tnet2.modules[1].modules[1].gradWeight:sum()')
    print(self.net2.modules[1].modules[2].gradWeight:sum()+
            self.net2.modules[1].modules[2].gradBias:sum()..
            '\tnet2.modules[1].modules[2].gradWeight:sum()')
    return loss, torch.cat({self.enc_dw,self.dec2_dw})
end

function minimal:getQUpdate()
    -- net1 forward
    for t = 1,3 do
        self:net1forward()
    end
end

function minimal:learnQminibatch()
    -- net2 fwd bwd
    for t = 1,3 do
        print(t)
        local input2 = {torch.rand(3,2),torch.rand(3,2)}
        self:net2forwardbackward(input2)
    end
end

local m = minimal.create()
m:getQUpdate()
m:learnQminibatch()
