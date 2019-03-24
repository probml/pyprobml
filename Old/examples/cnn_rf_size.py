#Compute input size that leads to a 1x1 output size, among other things   
#https://stackoverflow.com/questions/35582521/how-to-calculate-receptive-field-size

# [filter size, stride, padding]

convnet =[[11,4,0],[3,2,0],[5,1,2],[3,2,0],[3,1,1],[3,1,1],[3,1,1],[3,2,0],[6,1,0]]
#layer_name = ['conv1','pool1','conv2','pool2','conv3','conv4','conv5','pool5','fc6-conv']
imsize = 227

inc = [3,1,0] # inception block
noop = [1,1,0]
convnet =[[7,2,0], noop, noop, [3,1,0], noop, inc, inc, [3,2,0], 
    inc, inc, inc, inc, inc, [2,2,0]]
imsize = 96

def outFromIn(isz, layernum = 9, net = convnet):
    if layernum>len(net): layernum=len(net)

    totstride = 1
    insize = isz
    #for layerparams in net:
    for layer in range(layernum):
        fsize, stride, pad = net[layer]
        outsize = (insize - fsize + 2*pad) / stride + 1
        insize = outsize
        totstride = totstride * stride
    return outsize, totstride

def inFromOut( layernum = 9, net = convnet):
    if layernum>len(net): layernum=len(net)
    outsize = 1
    #for layerparams in net:
    for layer in reversed(range(layernum)):
        fsize, stride, pad = net[layer]
        outsize = ((outsize -1)* stride) + fsize
    RFsize = outsize
    return RFsize
    
print("layer output sizes given image = %dx%d" % (imsize, imsize))
for i in range(len(convnet)):
    p = outFromIn(imsize,i+1)
    rf = inFromOut(i+1)
    print("layer %d, rf %d" % (i+1, rf))