-- sharing weights + fwd/bwd



require 'nn'

-- Create networks
net1 = nn.Sequential()
net1:add(nn.Linear(2,2))
net1:add(nn.Linear(2,1))

net2 = nn.Sequential()
net2:add(nn.Linear(2,2))
net2:add(nn.Linear(2,2))

crit = nn.MSECriterion()

print('net1')
print(net1)
print('net2')
print(net2)

-- Get modules
enc = net1.modules[1]
dec1 = net1.modules[2]
dec2 = net2.modules[2]


-- Get Parameters
enc_w, enc_dw = enc:getParameters()
dec1_w, dec1_dw = dec1:getParameters()
dec2_w, dec2_dw = dec2:getParameters()


-- Share NOTE You may have to do a ParallelTable here instead
net2.modules[1]:share(enc,'weight','bias','gradWeight','gradBias')


-- Zero Grad Parameters
enc_dw:zero()
dec1_dw:zero()
dec2_dw:zero()


-- Data
input = torch.rand(3,2)
target1 = torch.rand(3,1)
target2 = torch.rand(3,2)


-- Forward on net1 NOTE may have to clone net1 as target net1
out1 = net1:forward(input)


-- Zero Grad Parameters
enc_dw:zero()
dec1_dw:zero()
dec2_dw:zero()


-- Inspect Grad Parameters
print('Before fwd/bwd: These should be equal')
print(enc_dw:sum()..'\tenc_dw:sum()')
print(net2.modules[1].gradWeight:sum()+net2.modules[1].gradBias:sum()..'\tnet2.modules[1].gradWeight:sum()')

-- Forward on net2
out2 = net2:forward(input)
loss = crit:forward(out2,target2)


-- Backward on net2
gout2 = crit:backward(out2, target2)
gin2 = net2:backward(input,gout2)

print('After fwd/bwd: These should be equal')
print(enc_dw:sum()..'\tenc_dw:sum()')
print(net2.modules[1].gradWeight:sum()+net2.modules[1].gradBias:sum()..'\tnet2.modules[1].gradWeight:sum()')
