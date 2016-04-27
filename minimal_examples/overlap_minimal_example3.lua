-- sharing weights + fwd/bwd + parallel + in functions

require 'nn'

-- Create networks
net1 = nn.Sequential()
net1:add(nn.Linear(2,2))
net1:add(nn.Linear(2,1))

net2 = nn.Sequential()
net2parallel = nn.ParallelTable()
net2parallel:add(nn.Linear(2,2))
net2parallel:add(nn.Linear(2,2))
net2:add(net2parallel)
net2:add(nn.CAddTable())
net2:add(nn.Linear(2,2))

crit = nn.MSECriterion()

print('net1')
print(net1)
print('net2')
print(net2)

-- Get modules
enc = net1.modules[1]
dec1 = net1.modules[2]
dec2 = net2.modules[3]


-- Get Parameters
enc_w, enc_dw = enc:getParameters()
dec1_w, dec1_dw = dec1:getParameters()
dec2_w, dec2_dw = dec2:getParameters()

-- Share NOTE You may have to do a ParallelTable here instead
net2.modules[1].modules[1]:share(enc,'weight','bias','gradWeight','gradBias')
net2.modules[1].modules[2]:share(enc,'weight','bias','gradWeight','gradBias')


-- Zero Grad Parameters
enc_dw:zero()
dec1_dw:zero()
dec2_dw:zero()

-- Functions
function net1forward()
    local input1 = torch.rand(3,2)

    -- Clone net1
    net1clone = net1:clone()
    local out1 = net1:forward(input1)
end

function net2forwardbackward(funcinput2)
    -- Zero Grad Parameters
    enc_dw:zero()
    dec1_dw:zero()
    dec2_dw:zero()


    -- Inspect Grad Parameters
    print('Before fwd/bwd: These should be equal')
    print(enc_dw:sum()..'\tenc_dw:sum()')
    print(net2.modules[1].modules[1].gradWeight:sum()+
            net2.modules[1].modules[1].gradBias:sum()..
            '\tnet2.modules[1].modules[1].gradWeight:sum()')
    print(net2.modules[1].modules[2].gradWeight:sum()+
            net2.modules[1].modules[2].gradBias:sum()..
            '\tnet2.modules[1].modules[2].gradWeight:sum()')


    -- Forward on net2
    local out2 = net2:forward(funcinput2)
    local loss = crit:forward(out2,funcinput2[2])


    -- Backward on net2
    local gout2 = crit:backward(out2, funcinput2[2])
    local gin2 = net2:backward(funcinput2,gout2)


    print('After fwd/bwd: These should be equal')
    print(enc_dw:sum()..'\tenc_dw:sum()')
    print(net2.modules[1].modules[1].gradWeight:sum()+
            net2.modules[1].modules[1].gradBias:sum()..
            '\tnet2.modules[1].modules[1].gradWeight:sum()')
    print(net2.modules[1].modules[2].gradWeight:sum()+
            net2.modules[1].modules[2].gradBias:sum()..
            '\tnet2.modules[1].modules[2].gradWeight:sum()')
    return loss, torch.cat({enc_dw,dec2_dw})
end

-- net1 forward
for t = 1,3 do
    net1forward()
end

-- net2 fwd bwd
for t = 1,3 do
    print(t)
    local input2 = {torch.rand(3,2),torch.rand(3,2)}
    net2forwardbackward(input2)
end
