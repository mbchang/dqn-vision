-- sharing weights + fwd/bwd + parallel

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
dec2 = net2.modules[2]


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


-- Data
input1 = torch.rand(3,2)
input2 = {torch.rand(3,2),torch.rand(3,2)}
target1 = torch.rand(3,1)
target2 = torch.rand(3,2)


-- Forward on net1
out1 = net1:forward(input1)


-- Clone net1
net1clone = net1:clone()


-- Zero Grad Parameters
enc_dw:zero()
dec1_dw:zero()
dec2_dw:zero()


-- Inspect Grad Parameters
print('Before fwd/bwd: These should be equal')
print(enc_dw:sum()..'\tenc_dw:sum()')
print(net2.modules[1].modules[1].gradWeight:sum()+net2.modules[1].modules[1].gradBias:sum()..'\tnet2.modules[1].modules[1].gradWeight:sum()')
print(net2.modules[1].modules[2].gradWeight:sum()+net2.modules[1].modules[2].gradBias:sum()..'\tnet2.modules[1].modules[2].gradWeight:sum()')


-- Forward on net2
out2 = net2:forward(input2)
loss = crit:forward(out2,target2)


-- Backward on net2
gout2 = crit:backward(out2, target2)
gin2 = net2:backward(input2,gout2)


print('After fwd/bwd: These should be equal')
print(enc_dw:sum()..'\tenc_dw:sum()')
print(net2.modules[1].modules[1].gradWeight:sum()+net2.modules[1].modules[1].gradBias:sum()..'\tnet2.modules[1].modules[1].gradWeight:sum()')
print(net2.modules[1].modules[2].gradWeight:sum()+net2.modules[1].modules[2].gradBias:sum()..'\tnet2.modules[1].modules[2].gradWeight:sum()')
