require 'torch'
require 'nn'
require 'image'
require 'paths'
require 'lib/NonparametricPatchAutoencoderFactory'
require 'lib/MaxCoord'
require 'lib/utils'
require 'lib/AdaptiveInstanceNormalization'
require 'nngraph'
require 'cudnn'
require 'cunn'

local matio = require 'matio'
local cmd = torch.CmdLine()

cmd:option('-style', 'input/portrait_10.jpg', 'path to the style image')
cmd:option('-content', 'input/13960.jpg', 'path to the content image')
cmd:option('-alpha', 0.6)
cmd:option('-patchSize', 3)
cmd:option('-patchStride', 1)
cmd:option('-vgg1', 'models/conv1_1.t7', 'Path to the VGG conv1_1')
cmd:option('-vgg2', 'models/conv2_1.t7', 'Path to the VGG conv2_1')
cmd:option('-vgg3', 'models/conv3_1.t7', 'Path to the VGG conv3_1')
cmd:option('-vgg4', 'models/conv4_1.t7', 'Path to the VGG conv4_1')
cmd:option('-vgg5', 'models/conv5_1.t7', 'Path to the VGG conv5_1')
cmd:option('-decoder5', 'models/dec5_1.t7', 'Path to the decoder5')
cmd:option('-decoder4', 'models/dec4_1.t7', 'Path to the decoder4')
cmd:option('-decoder3', 'models/dec3_1.t7', 'Path to the decoder3')
cmd:option('-decoder2', 'models/dec2_1.t7', 'Path to the decoder2')
cmd:option('-decoder1', 'models/dec1_1.t7', 'Path to the decoder1')

cmd:option('-contentSize', 768, 'New (minimum) size for the content image, keeping the original size if set to 0')
cmd:option('-styleSize', 768, 'New (minimum) size for the style image, keeping the original size if set to 0')
cmd:option('-outputDir', 'output/alley_1', 'Directory to save the output image(s)')

opt = cmd:parse(arg)

--/////////////////////////////////////////////////////////////////////
function loadModel()
    vgg1 = torch.load(opt.vgg1)
    vgg2 = torch.load(opt.vgg2)
    vgg3 = torch.load(opt.vgg3)
    vgg4 = torch.load(opt.vgg4)
    vgg5 = torch.load(opt.vgg5)
    
    decoder5 = torch.load(opt.decoder5)
    decoder4 = torch.load(opt.decoder4)
    decoder3 = torch.load(opt.decoder3)
    decoder2 = torch.load(opt.decoder2)
    decoder1 = torch.load(opt.decoder1)

    adain5 = nn.AdaptiveInstanceNormalization(vgg5:get(#vgg5-1).nOutputPlane)
    adain4 = nn.AdaptiveInstanceNormalization(vgg4:get(#vgg4-1).nOutputPlane)
    adain3 = nn.AdaptiveInstanceNormalization(vgg3:get(#vgg3-1).nOutputPlane)
    adain2 = nn.AdaptiveInstanceNormalization(vgg2:get(#vgg2-1).nOutputPlane)
    adain1 = nn.AdaptiveInstanceNormalization(vgg1:get(#vgg1-1).nOutputPlane)

    print('GPU mode')
    vgg1:cuda()
    vgg2:cuda()
    vgg3:cuda()
    vgg4:cuda()
    vgg5:cuda()
    
    adain5:cuda()
	adain4:cuda()
	adain3:cuda()
	adain2:cuda()
	adain1:cuda()
    
    decoder1:cuda()
    decoder2:cuda()
    decoder3:cuda()
    decoder4:cuda()
    decoder5:cuda()
end

--/////////////////////////////////////////////////////////////////////
function normalize_features(x)
    local x2 = torch.pow(x, 2)
    local sum_x2 = torch.sum(x2, 1)
    local dis_x2 = torch.sqrt(sum_x2)
    local Nx = torch.cdiv(x, dis_x2:expandAs(x) + 1e-8)
    -- local Nx = torch.cdiv(x, dis_x2:expandAs(x))
    dis_x2 = (dis_x2-torch.min(dis_x2))/(torch.max(dis_x2)-torch.min(dis_x2))
    return Nx,dis_x2
end

--////////////////////////////////////////////////////////////////////
function whitenMatrix(featureIn)
    local feature = featureIn:clone() -- c x hw
    local sz = feature:size()
    local ft_mean = torch.mean(feature,2)
    feature = feature - ft_mean:expandAs(feature)
    local ft_std = torch.std(feature,2)
    local ft_conv = torch.mm(feature,feature:t()):div(sz[2]-1)
    local u,e,v = torch.svd(ft_conv:float(),'A')
    local k_c = sz[1]
    for i=1,sz[1] do
        if e[i]<0.00001 then
            k_c = i-1
            break
        end
    end
    local d = e[{{1,k_c}}]:sqrt():pow(-1)
    local m = (v[{{},{1,k_c}}]:cuda())*torch.diag(d:cuda())*(v[{{},{1,k_c}}]:t():cuda())
    return m:cuda(),ft_mean:cuda(),ft_std:cuda()
end

--///////////////////////////////////////////////////////////////////
function colorMatrix(featureIn)
    local feature = featureIn:clone()
    local sz = feature:size()
    local ft_mean = torch.mean(feature,2)
    feature = feature - ft_mean:expandAs(feature)
    local ft_std = torch.std(feature,2)
    local ft_conv = torch.mm(feature,feature:t()):div(sz[2]-1)
    local u,e,v = torch.svd(ft_conv:float(),'A')
    local k_c = sz[1]
    for i=1,sz[1] do
        if e[i]<0.00001 then
            k_c = i-1
            break
        end
    end
    local d = e[{{1,k_c}}]:sqrt()
    local m = (v[{{},{1,k_c}}]:cuda())*torch.diag(d:cuda())*(v[{{},{1,k_c}}]:t():cuda())
    return m:cuda(),ft_mean:cuda(),ft_std:cuda()
end

--////////////////////////////////////////////////////////////////////
function sqrtInvMatrix(mtx) -- cxc
    local sz = mtx:size()
    local u,e,v = torch.svd(mtx:float(),'A')
    local k_c = sz[1]
    for i=1,sz[1] do
        if e[i]<0.00001 then
            k_c = i-1
            break
        end
    end
    local d = e[{{1,k_c}}]:sqrt():pow(-1)
    local m = (v[{{},{1,k_c}}]:cuda())*torch.diag(d:cuda())*(v[{{},{1,k_c}}]:t():cuda())
    return m:cuda()
end
--//////////////////////////////////////////////////////////////////
function invMatrix(mtx) -- cxc
    local sz = mtx:size()
    local u,e,v = torch.svd(mtx:float(),'A')
    local k_c = sz[1]
    for i=1,sz[1] do
        if e[i]<0.00001 then
            k_c = i-1
            break
        end
    end
    local d = e[{{1,k_c}}]:pow(-1)
    local m = (v[{{},{1,k_c}}]:cuda())*torch.diag(d:cuda())*(v[{{},{1,k_c}}]:t():cuda())
    return m:cuda()
end

--///////////////////////////////////////////////////////////////////
function sqrtMatrix(mtx)
    local sz = mtx:size()
    local u,e,v = torch.svd(mtx:float(),'A')
    local k_c = sz[1]
    for i=1,sz[1] do
        if e[i]<0.00001 then
            k_c = i-1
            break
        end
    end
    local d = e[{{1,k_c}}]:sqrt()
    local m = (v[{{},{1,k_c}}]:cuda())*torch.diag(d:cuda())*(v[{{},{1,k_c}}]:t():cuda())
    return m:cuda()
end

--/////////////////////////////////////////////////////////////////////////////////////
function feature_swap(contentFeature, styleFeature)
    
    local sg = contentFeature:size()
    local contentFeature1 = contentFeature:view(sg[1], sg[2]*sg[3])
    local c_mean = torch.mean(contentFeature1, 2)
    contentFeature1 = contentFeature1 - c_mean:expandAs(contentFeature1)
    local contentCov = torch.mm(contentFeature1, contentFeature1:t()):div(sg[2]*sg[3]-1)  
    local c_u, c_e, c_v = torch.svd(contentCov:float(), 'A')  
    local k_c = sg[1]
    for i=1, sg[1] do
       if c_e[i] < 0.00001 then
            k_c = i-1
            break
       end
    end
    
    local sz = styleFeature:size()
    local styleFeature1 = styleFeature:view(sz[1], sz[2]*sz[3])
    local s_mean = torch.mean(styleFeature1, 2)
    styleFeature1 = styleFeature1 - s_mean:expandAs(styleFeature1)
    local styleCov = torch.mm(styleFeature1, styleFeature1:t()):div(sz[2]*sz[3]-1)  
    local s_u, s_e, s_v = torch.svd(styleCov:float(), 'A')
    local k_s = sz[1]
    for i=1, sz[1] do
       if s_e[i] < 0.00001 then
            k_s = i-1
            break
       end
    end

    local s_d = torch.sqrt(s_e[{{1,k_s}}]):pow(-1)
    local whiten_styleFeature = nil
    whiten_styleFeature = (s_v[{{},{1,k_s}}]:cuda()) * torch.diag(s_d:cuda()) * (s_v[{{},{1,k_s}}]:t():cuda()) * styleFeature1 
    local swap_enc, swap_dec = NonparametricPatchAutoencoderFactory.buildAutoencoder(whiten_styleFeature:resize(sz[1], sz[2], sz[3]), opt.patchSize, opt.patchStride, false, false, true)
    local swap = nn.Sequential()
    swap:add(swap_enc)
    swap:add(nn.MaxCoord())
    swap:add(swap_dec)
    swap:evaluate()
    swap:cuda()
    local c_d = torch.sqrt(c_e[{{1,k_c}}]):pow(-1)
    local s_d1 = torch.sqrt(s_e[{{1,k_s}}])
    local whiten_contentFeature = nil
    local targetFeature = nil
    whiten_contentFeature = (c_v[{{},{1,k_c}}]:cuda()) * torch.diag(c_d:cuda()) * (c_v[{{},{1,k_c}}]:t():cuda()) *contentFeature1
    local swap_latent = swap:forward(whiten_contentFeature:resize(sg[1], sg[2], sg[3])):clone()
    local swap_latent1 = swap_latent:view(sg[1], sg[2]*sg[3])
    targetFeature = (s_v[{{},{1,k_s}}]:cuda()) * (torch.diag(s_d1:cuda())) * (s_v[{{},{1,k_s}}]:t():cuda()) * swap_latent1
    targetFeature = targetFeature + s_mean:expandAs(targetFeature)
    local tFeature = targetFeature:resize(sg[1], sg[2], sg[3])
    return tFeature
end

--///////////////////////////////////////////////////////////////////////
function feature_wct(contentFeature, styleFeature)
   
    local sg = contentFeature:size()
    local contentFeature1 = contentFeature:view(sg[1], sg[2]*sg[3])
    local c_mean = torch.mean(contentFeature1, 2)
    contentFeature1 = contentFeature1 - c_mean:expandAs(contentFeature1)
    local contentCov = torch.mm(contentFeature1, contentFeature1:t()):div(sg[2]*sg[3]-1) 
    local c_u, c_e, c_v = torch.svd(contentCov:float(), 'A')  
    local k_c = sg[1]
    
    for i=1, sg[1] do
       if c_e[i] < 0.00001 then
            k_c = i-1
            break
       end
    end

    --k_c = sg[1]

    local sz = styleFeature:size()
    local styleFeature1 = styleFeature:view(sz[1], sz[2]*sz[3])
    local s_mean = torch.mean(styleFeature1, 2)
    styleFeature1 = styleFeature1 - s_mean:expandAs(styleFeature1)
    local styleCov = torch.mm(styleFeature1, styleFeature1:t()):div(sz[2]*sz[3]-1) 
    local s_u, s_e, s_v = torch.svd(styleCov:float(), 'A')
    local k_s = sz[1]
    for i=1, sz[1] do
       if s_e[i] < 0.00001 then
            k_s = i-1
            break
       end
    end

    --k_s = sz[1]

    --[[
    curQ = nil
    if cur_idx == 1 then
      curQ = Q.Q1
    elseif cur_idx == 2 then
      curQ = Q.Q2
    elseif cur_idx == 3 then
      curQ = Q.Q3
    elseif cur_idx == 4 then
      curQ = Q.Q4
    elseif cur_idx == 5 then
      curQ = Q.Q5
    end
    curQ = curQ:cuda()
    print('current idx = '.. tostring(cur_idx))
    cur_idx = cur_idx-1
    --]]    

    local c_d = c_e[{{1,k_c}}]:sqrt():pow(-1)
    local s_d1 = s_e[{{1,k_s}}]:sqrt()
    local whiten_contentFeature = nil
    local targetFeature = nil
    
    -- ZCA
    whiten_contentFeature = (c_v[{{},{1,k_c}}]:cuda()) * torch.diag(c_d:cuda()) * (c_v[{{},{1,k_c}}]:t():cuda()) *contentFeature1      
    
    -- PCA
    --whiten_contentFeature = torch.diag(c_d:cuda()) * (c_v[{{},{1,k_c}}]:t():cuda()) *contentFeature1

    -- Cholesky
    --[[
    local chol_d = c_e[{{1,k_c}}]:pow(-1)
    whiten_M = (c_v[{{},{1,k_c}}]:cuda()) * (torch.diag(chol_d:cuda()) * (c_v[{{},{1,k_c}}]:t():cuda()))
    whiten_M = whiten_M:float()
    whiten_M = torch.potrf(whiten_M,'L')
    whiten_M = whiten_M:t():cuda()
    whiten_contentFeature = whiten_M * contentFeature1 -- CxN
    --]]

    -- ZCA cor and PCA cor
    --[[
    V_std = torch.std(contentFeature1,2):squeeze()
    V_sqrt = torch.diag(V_std)
    V_sqrt_inv = invMatrix(V_sqrt)
    V_sqrt_inv = V_sqrt_inv:float()
    P = V_sqrt_inv * contentCov:float() * V_sqrt_inv
    G, Theta, Gt = torch.svd(P, 'A')      
    G_d = Theta[{{1,k_c}}]:sqrt():pow(-1)
    whiten_M =  (G[{{},{1,k_c}}]:cuda()) * torch.diag(G_d:cuda()) * (G[{{},{1,k_c}}]:t():cuda()) * V_sqrt:cuda()
    --whiten_M = torch.diag(G_d:cuda()) * (G[{{},{1,k_c}}]:t():cuda()) * V_sqrt:cuda()
    whiten_contentFeature = whiten_M * contentFeature1 -- CxN   
    --]]

    --whiten_contentFeature = curQ*whiten_contentFeature
    
    targetFeature = (s_v[{{},{1,k_s}}]:cuda()) * (torch.diag(s_d1:cuda())) * (s_v[{{},{1,k_s}}]:t():cuda()) * whiten_contentFeature
    targetFeature = targetFeature + s_mean:expandAs(targetFeature)
    local tFeature = targetFeature:resize(sg[1], sg[2], sg[3])
    return tFeature
end

--////////////////////////////////////////////////////////////////////
function feature_mk(contentFeature, styleFeature)
    
    local eps=1e-10
    local cDim = contentFeature:size()
    local contentFeature1 = contentFeature:view(cDim[1], cDim[2]*cDim[3]) -- cxhw
    local c_mean = torch.mean(contentFeature1, 2) 
    contentFeature1 = contentFeature1 - c_mean:expandAs(contentFeature1)
    local contentCov = torch.mm(contentFeature1, contentFeature1:t()):div(cDim[2]*cDim[3]-1) -- cxc
        
    local sDim = styleFeature:size()
    local styleFeature1 = styleFeature:view(sDim[1], sDim[2]*sDim[3]) -- cxhw
    local s_mean = torch.mean(styleFeature1, 2)
    styleFeature1 = styleFeature1 - s_mean:expandAs(styleFeature1)
    local styleCov = torch.mm(styleFeature1, styleFeature1:t()):div(sDim[2]*sDim[3]-1) -- cxc
    
    local Da2,Ua = torch.eig(contentCov:float(),'V') -- return e(mx2),V(mxm)
    Ua = Ua:t()
    Da2 = Da2[{{},{1}}]:squeeze():cuda()
    Da2 = torch.diag(Da2)
    Da2[torch.lt(Da2,0)] = 0    
    Da2 = Da2+eps
    local Da = Da2:sqrt():cuda() -- cxc
    Ua = Ua:cuda()
    
    styleCov = styleCov:cuda()
    local C = Da*Ua:t()*styleCov*Ua*Da
    
    local Dc2,Uc = torch.eig(C:float(),'V') -- return e,V
    Uc = Uc:t()
    Dc2 = Dc2[{{},{1}}]:squeeze():cuda()
    Dc2 = torch.diag(Dc2)
    Dc2[torch.lt(Dc2,0)] = 0
    Dc2 = Dc2+eps
    local Dc = Dc2:sqrt()
    Uc = Uc:cuda()    

    local Da_inv = Da:pow(-1)

    local T = Ua*Da_inv*Uc*Dc*Uc:t()*Da_inv*Ua:t() -- cxc
    
    local targetFeature = T*contentFeature1
    targetFeature = targetFeature + s_mean:expandAs(targetFeature)
    local resFeature = targetFeature:resize(cDim[1],cDim[2],cDim[3])
    return resFeature
end

--////////////////////////////////////////////////////////////////////
function feature_mk2(contentFeature, styleFeature)
    
    local eps=1e-10
    local cDim = contentFeature:size()
    local contentFeature1 = contentFeature:view(cDim[1], cDim[2]*cDim[3]) -- cxhw
    local c_mean = torch.mean(contentFeature1, 2) 
    contentFeature1 = contentFeature1 - c_mean:expandAs(contentFeature1)
    local contentCov = torch.mm(contentFeature1, contentFeature1:t()):div(cDim[2]*cDim[3]-1) -- cxc
        
    local sDim = styleFeature:size()
    local styleFeature1 = styleFeature:view(sDim[1], sDim[2]*sDim[3]) -- cxhw
    local s_mean = torch.mean(styleFeature1, 2)
    styleFeature1 = styleFeature1 - s_mean:expandAs(styleFeature1)
    local styleCov = torch.mm(styleFeature1, styleFeature1:t()):div(sDim[2]*sDim[3]-1) -- cxc
   
    local sqrtInvU = sqrtInvMatrix(contentCov)
    local sqrtU = sqrtMatrix(contentCov)
    local C = sqrtU*styleCov*sqrtU
    local sqrtC = sqrtMatrix(C)
    local T = sqrtInvU*sqrtC*sqrtInvU    
    local targetFeature = T*contentFeature1
    targetFeature = targetFeature + s_mean:expandAs(targetFeature)
    local resFeature = targetFeature:resize(cDim[1],cDim[2],cDim[3])
    return resFeature
end
--//////////////////////////////////////////////////////////////////
function feature_mk3(contentFeature, styleFeature)
    
    local eps=1e-10
    local cDim = contentFeature:size() -- cxN
    local contentFeature1 = contentFeature -- cxN
    local c_mean = torch.mean(contentFeature1, 2) 
    contentFeature1 = contentFeature1 - c_mean:expandAs(contentFeature1)
    local contentCov = torch.mm(contentFeature1, contentFeature1:t()):div(cDim[2]-1) -- cxc
        
    local sDim = styleFeature:size() -- cxN
    local styleFeature1 = styleFeature -- cxN
    local s_mean = torch.mean(styleFeature1, 2)
    styleFeature1 = styleFeature1 - s_mean:expandAs(styleFeature1)
    local styleCov = torch.mm(styleFeature1, styleFeature1:t()):div(sDim[2]-1) -- cxc
   
    local sqrtInvU = sqrtInvMatrix(contentCov)
    local sqrtU = sqrtMatrix(contentCov)
    local C = sqrtU*styleCov*sqrtU
    local sqrtC = sqrtMatrix(C)
    local T = sqrtInvU*sqrtC*sqrtInvU    
    local targetFeature = T*contentFeature1
    targetFeature = targetFeature + s_mean:expandAs(targetFeature)
    local resFeature = targetFeature
    return resFeature -- cxN

end

function feature_mk3_sem(contentFeature, styleFeature,maskC,maskS)

    local eps=1e-10

    maskC = maskC:cuda()
    maskS = maskS:cuda()

    local cDim = contentFeature:size()
    local contentFeature1 = contentFeature:view(cDim[1], cDim[2]*cDim[3]) -- cxhw
    local sDim = styleFeature:size()
    local styleFeature1 = styleFeature:view(sDim[1], sDim[2]*sDim[3]) -- cxhw
    
    local cView = maskC:view(-1)
    local sView = maskS:view(-1)
    
    local targetFeature1 = contentFeature1:clone():zero()

    for k=1,5 do
      local cFg = torch.LongTensor(torch.find(cView,k-1)) 
      local sFg = torch.LongTensor(torch.find(sView,k-1))
      local cFt = contentFeature1:index(2,cFg):view(cDim[1],cFg:nElement())
      local sFt = styleFeature1:index(2,sFg):view(sDim[1],sFg:nElement())
      local tFt = feature_mk3(cFt,sFt)
      targetFeature1:indexCopy(2,cFg,tFt)
    end

    targetFeature1 = targetFeature1:viewAs(contentFeature)
    return targetFeature1
end

--///////////////////////////////////////////////////////////////////
function feature_clamp(contentFeature,styleFeature)

    -- check feature
    --[[
    local cFt = contentFeature[{{1},{},{}}]:squeeze()
    local sFt = styleFeature[{{1},{},{}}]:squeeze()
    local disp = torch.cat(cFt,sFt)
    image.display(disp)
    --]]

    local sz_c = contentFeature:size()
    local sz_s = styleFeature:size()
    local contentFeatureView = contentFeature:view(sz_c[1],sz_c[2]*sz_c[3])
    local styleFeatureView = styleFeature:view(sz_s[1],sz_s[2]*sz_s[3])
    local cWhitenM,cWhitenMean,cWhitenStd = whitenMatrix(contentFeatureView)
    local sWhitenM,sWhitenMean,sWhitenStd = whitenMatrix(styleFeatureView)
    local sColorM,sColorMean,sColorStd = colorMatrix(styleFeatureView)
    -- whiten 
    local contentWhiten = cWhitenM*(contentFeatureView-cWhitenMean:expandAs(contentFeatureView))
    local styleWhiten = sWhitenM*(styleFeatureView-sWhitenMean:expandAs(styleFeatureView))
    contentWhiten = contentWhiten:view(sz_c[1],sz_c[2],sz_c[3])
    styleWhiten = styleWhiten:view(sz_s[1],sz_s[2],sz_s[3])
    -- blend
    local gainMap = torch.cdiv(styleWhiten,contentWhiten)
    gainMap = torch.clamp(gainMap,0.5,1.0)
    local contentRemap = torch.cmul(contentWhiten,gainMap)
    contentRemap = contentRemap:view(sz_c[1],sz_c[2]*sz_c[3])
    contentRemap = sColorM*contentRemap+sColorMean:expandAs(contentRemap)
    contentRemap = contentRemap:view(sz_c[1],sz_c[2],sz_c[3])
    return contentRemap
end

--/////////////////////////////////////////////////////////////////////////
function feature_blend(contentFeature,styleFeature,alpha)
    local szC = contentFeature:size()
    local szS = styleFeature:size()
    local contentFtView = contentFeature:view(szC[1],szC[2]*szC[3])
    local styleFtView = styleFeature:view(szS[1],szS[2]*szS[3])
    local contentFtN,contentFtD = normalize_features(contentFtView)
    local styleFtN,styleFtD = normalize_features(styleFtView)
    
    contentFtD = contentFtD - 0.05
    contentFtD[contentFtD:lt(0.000001)] = 0.0
    contentFtD[contentFtD:gt(0.000001)] = 1.0
    local gainMap = contentFtD*alpha
    gainMap = gainMap:view(1,szC[2],szC[3])
    --image.display(gainMap:squeeze())
    gainMap = gainMap:expandAs(contentFeature)

    --[[
    contentFtD = -300.0*(contentFtD-0.05)
    local gainMap = torch.cinv((1+torch.exp(contentFtD)))
    gainMap = gainMap:view(1,szC[2],szC[3])
    image.display(gainMap:squeeze())
    gainMap = gainMap:expandAs(contentFeature)
    gainMap = alpha*gainMap 
    --]]

    return torch.cmul(contentFeature,gainMap)+torch.cmul(styleFeature,1-gainMap)
end

--///////////////////////////////////////////////////////////////////////////////////////////
local function styleTransfer_wct(content, style)

    loadModel()
    
    print('Start wct')

    content = content:cuda()
    style = style:cuda() 
    local cF5 = vgg5:forward(content):clone()
    local sF5 = vgg5:forward(style):clone()
    vgg5 = nil
    local csF5 = nil
    --csF5 = feature_swap(cF5, sF5)
    csF5 = feature_wct(cF5, sF5)
    csF5 = opt.alpha * csF5 + (1.0-opt.alpha) * cF5
    local Im5 = decoder5:forward(csF5)
    decoder5 = nil

    local cF4 = vgg4:forward(Im5):clone()
    local sF4 = vgg4:forward(style):clone()
    vgg4 = nil
    --local csF4 = feature_swap(cF4,sF4)
    local csF4 = feature_wct(cF4, sF4)
    csF4 = opt.alpha * csF4 + (1.0-opt.alpha) * cF4
    local Im4 = decoder4:forward(csF4)
    decoder4 = nil

    local cF3 = vgg3:forward(Im4):clone()
    local sF3 = vgg3:forward(style):clone()
    vgg3 = nil
    local csF3 = feature_wct(cF3, sF3)
    csF3 = opt.alpha * csF3 + (1.0-opt.alpha) * cF3 

    local Im3 = decoder3:forward(csF3)
    decoder3 = nil
    local cF2 = vgg2:forward(Im3):clone()
    local sF2 = vgg2:forward(style):clone()
    vgg2 = nil

    local csF2 = feature_wct(cF2, sF2)
    csF2 = opt.alpha * csF2 + (1.0-opt.alpha) * cF2
    local Im2 = decoder2:forward(csF2)
    decoder2 = nil

    local cF1 = vgg1:forward(Im2):clone()
    local sF1 = vgg1:forward(style):clone()
    vgg1 = nil
    local csF1 = feature_wct(cF1, sF1)
    csF1 = opt.alpha * csF1 + (1.0-opt.alpha) * cF1
    local Im1 = decoder1:forward(csF1) 
    decoder1 = nil
    return Im1
end

--////////////////////////////////////////////////////////////////////////////////////
local function styleTransfer_adaIn(content, style)
    loadModel()
     
    print('Start AdaIn')

    content = content:cuda()
    style = style:cuda()

    local cF5 = vgg5:forward(content):clone()
    local sF5 = vgg5:forward(style):clone()
    vgg5 = nil
    csF5 = adain5:forward({cF5, sF5}):squeeze()
    csF5 = opt.alpha * csF5 + (1.0-opt.alpha) * cF5
    local Im5 = decoder5:forward(csF5)
    decoder5 = nil

    local cF4 = vgg4:forward(Im5):clone()
    local sF4 = vgg4:forward(style):clone()
    vgg4 = nil
    local csF4 = adain4:forward({cF4, sF4}):squeeze()
    csF4 = opt.alpha * csF4 + (1.0-opt.alpha) * cF4
    local Im4 = decoder4:forward(csF4)
    decoder4 = nil

    local cF3 = vgg3:forward(Im4):clone()
    local sF3 = vgg3:forward(style):clone()
    vgg3 = nil
    local csF3 = adain3:forward({cF3, sF3}):squeeze()
    csF3 = opt.alpha * csF3 + (1.0-opt.alpha) * cF3 
    local Im3 = decoder3:forward(csF3)
    decoder3 = nil

    local cF2 = vgg2:forward(Im3):clone()
    local sF2 = vgg2:forward(style):clone()
    vgg2 = nil
    local csF2 = adain2:forward({cF2, sF2}):squeeze()
    csF2 = opt.alpha * csF2 + (1.0-opt.alpha) * cF2
    local Im2 = decoder2:forward(csF2)
    decoder2 = nil

    local cF1 = vgg1:forward(Im2):clone()
    local sF1 = vgg1:forward(style):clone()
    vgg1 = nil
    local csF1 = adain1:forward({cF1, sF1}):squeeze()
    csF1 = opt.alpha * csF1 + (1.0-opt.alpha) * cF1
    local Im1 = decoder1:forward(csF1) 
    decoder1 = nil
    return Im1
end

--//////////////////////////////////////////////////////////////////////////////////////
local function styleTransfer_clamp(content, style)
    
    loadModel()
    
    local cSz = content:size()
    local sSz = style:size()
    
    content = content:cuda()
    style = style:cuda()

    --[[
    local cF5 = vgg5:forward(content):clone()
    local sF5 = vgg5:forward(style):clone()
    vgg5 = nil
    csF5 = feature_clamp(cF5,sF5)
    csF5 = opt.alpha * csF5 + (1.0-opt.alpha) * cF5
    local Im5 = decoder5:forward(csF5):clone()
    decoder5 = nil
    --]]

    vgg5 = nil
    decoder5 = nil
    local Im5 = content

    Im5 = image.scale(Im5:float(),cSz[3],cSz[2])
    Im5 = Im5:cuda()
    local cF4 = vgg4:forward(Im5):clone()
    local sF4 = vgg4:forward(style):clone()
    vgg4 = nil
    --local csF4 = feature_clamp(cF4,sF4)
    local csF4 = feature_blend(cF4,sF4,0.8)
    csF4 = opt.alpha * csF4 + (1.0-opt.alpha) * cF4
    local Im4 = decoder4:forward(csF4):clone()
    decoder4 = nil

    Im4 = image.scale(Im4:float(),cSz[3],cSz[2])
    Im4 = Im4:cuda()
    local cF3 = vgg3:forward(Im4):clone()
    local sF3 = vgg3:forward(style):clone()
    vgg3 = nil
    --local csF3 = feature_clamp(cF3,sF3)
    local csF3 = feature_blend(cF3,sF3,0.7)
    csF3 = opt.alpha * csF3 + (1.0-opt.alpha) * cF3 
    local Im3 = decoder3:forward(csF3):clone()
    decoder3 = nil

    Im3 = image.scale(Im3:float(),cSz[3],cSz[2])
    Im3 = Im3:cuda()
    local cF2 = vgg2:forward(Im3):clone()
    local sF2 = vgg2:forward(style):clone()
    vgg2 = nil
    --local csF2 = feature_clamp(cF2,sF2)
    local csF2 = feature_blend(cF2,sF2,0.6)
    csF2 = opt.alpha * csF2 + (1.0-opt.alpha) * cF2
    local Im2 = decoder2:forward(csF2):clone()
    decoder2 = nil

    Im2 = image.scale(Im2:float(),cSz[3],cSz[2])
    Im2 = Im2:cuda()
    local cF1 = vgg1:forward(Im2):clone()
    local sF1 = vgg1:forward(style):clone()
    vgg1 = nil
    --local csF1 = feature_clamp(cF1,sF1)
    local csF1 = feature_blend(cF1,sF1,0.3)
    csF1 = opt.alpha * csF1 + (1.0-opt.alpha) * cF1
    local Im1 = decoder1:forward(csF1):clone()
    decoder1 = nil
    Im1 = image.scale(Im1:float(),cSz[3],cSz[2])
    Im1 = Im1:cuda()

    return Im1
end

--///////////////////////////////////////////////////////////////////////////////////////
local function styleTransfer_mk(content, style)

    loadModel()
    
    print('Start MK')

    content = content:cuda()
    style = style:cuda() 
    local cF5 = vgg5:forward(content):clone()
    local sF5 = vgg5:forward(style):clone()
    vgg5 = nil
    local csF5 = nil
    csF5 = feature_mk2(cF5, sF5)
    csF5 = opt.alpha * csF5 + (1.0-opt.alpha) * cF5
    local Im5 = decoder5:forward(csF5)
    decoder5 = nil

    local cF4 = vgg4:forward(Im5):clone()
    local sF4 = vgg4:forward(style):clone()
    vgg4 = nil
    local csF4 = feature_mk2(cF4, sF4)
    csF4 = opt.alpha * csF4 + (1.0-opt.alpha) * cF4
    local Im4 = decoder4:forward(csF4)
    decoder4 = nil

    local cF3 = vgg3:forward(Im4):clone()
    local sF3 = vgg3:forward(style):clone()
    vgg3 = nil
    local csF3 = feature_mk2(cF3, sF3)
    csF3 = opt.alpha * csF3 + (1.0-opt.alpha) * cF3 

    local Im3 = decoder3:forward(csF3)
    decoder3 = nil
    local cF2 = vgg2:forward(Im3):clone()
    local sF2 = vgg2:forward(style):clone()
    vgg2 = nil

    local csF2 = feature_mk2(cF2, sF2)
    csF2 = opt.alpha * csF2 + (1.0-opt.alpha) * cF2
    local Im2 = decoder2:forward(csF2)
    decoder2 = nil

    local cF1 = vgg1:forward(Im2):clone()
    local sF1 = vgg1:forward(style):clone()
    vgg1 = nil
    local csF1 = feature_mk2(cF1, sF1)
    csF1 = opt.alpha * csF1 + (1.0-opt.alpha) * cF1
    local Im1 = decoder1:forward(csF1) 
    decoder1 = nil
    return Im1
end
--///////////////////////////////////////////////////////////////////////////////////////
local function styleTransfer_mk_sem(content, style)

    loadModel()
    content = content:cuda()
    style = style:cuda() 
    
    --///////
    local cF5 = vgg5:forward(content):clone()
    local sF5 = vgg5:forward(style):clone()
    vgg5 = nil
    local maskC = image.scale(masks.cMask,cF5:size(3),cF5:size(2),'simple')
    local maskS = image.scale(masks.sMask,sF5:size(3),sF5:size(2),'simple')
    local csF5 = nil
    csF5 = feature_mk3_sem(cF5, sF5,maskC,maskS)
    csF5 = opt.alpha * csF5 + (1.0-opt.alpha) * cF5
    local Im5 = decoder5:forward(csF5)
    decoder5 = nil

    --//////
    local cF4 = vgg4:forward(Im5):clone()
    local sF4 = vgg4:forward(style):clone()
    vgg4 = nil
    maskC = image.scale(masks.cMask,cF4:size(3),cF4:size(2),'simple')
    maskS = image.scale(masks.sMask,sF4:size(3),sF4:size(2),'simple')
    local csF4 = feature_mk3_sem(cF4, sF4,maskC,maskS)
    csF4 = opt.alpha * csF4 + (1.0-opt.alpha) * cF4
    local Im4 = decoder4:forward(csF4)
    decoder4 = nil

    --//////
    local cF3 = vgg3:forward(Im4):clone()
    local sF3 = vgg3:forward(style):clone()
    vgg3 = nil
    maskC = image.scale(masks.cMask,cF3:size(3),cF3:size(2),'simple')
    maskS = image.scale(masks.sMask,sF3:size(3),sF3:size(2),'simple')
    local csF3 = feature_mk3_sem(cF3, sF3,maskC,maskS)
    csF3 = opt.alpha * csF3 + (1.0-opt.alpha) * cF3 
    local Im3 = decoder3:forward(csF3)
    decoder3 = nil

    --///////
    local cF2 = vgg2:forward(Im3):clone()
    local sF2 = vgg2:forward(style):clone()
    vgg2 = nil
    maskC = image.scale(masks.cMask,cF2:size(3),cF2:size(2),'simple')
    maskS = image.scale(masks.sMask,sF2:size(3),sF2:size(2),'simple')
    local csF2 = feature_mk3_sem(cF2, sF2,maskC,maskS)
    csF2 = opt.alpha * csF2 + (1.0-opt.alpha) * cF2
    local Im2 = decoder2:forward(csF2)
    decoder2 = nil

    --//////
    local cF1 = vgg1:forward(Im2):clone()
    local sF1 = vgg1:forward(style):clone()
    vgg1 = nil
    maskC = image.scale(masks.cMask,cF1:size(3),cF1:size(2),'simple')
    maskS = image.scale(masks.sMask,sF1:size(3),sF1:size(2),'simple')
    local csF1 = feature_mk3_sem(cF1, sF1,maskC,maskS)
    csF1 = opt.alpha * csF1 + (1.0-opt.alpha) * cF1
    local Im1 = decoder1:forward(csF1) 
    decoder1 = nil

    return Im1
end

--////////////////////////////////////////////////////////////////////////////////

print('Creating save folder at ' .. opt.outputDir)
paths.mkdir(opt.outputDir)
local contentPaths = extractImageNamesRecursive(opt.content)
local stylePaths = extractImageNamesRecursive(opt.style)
print('Num of content images = ' .. tostring(#contentPaths))
print('Num of style images = ' .. tostring(#stylePaths))
for ck,cv in pairs(contentPaths) do
    for sk,sv in pairs(stylePaths) do
        local contentPath = cv
        local contentExt = paths.extname(contentPath)
        local contentName = paths.basename(contentPath,contentExt)
        local contentImg = image.load(contentPath, 3, 'float')
        contentImg = sizePreprocess(contentImg, opt.contentSize)
    
        local stylePath = sv
        local styleExt = paths.extname(stylePath)
        local styleName = paths.basename(stylePath,styleExt)
        local styleImg = image.load(stylePath, 3, 'float')
        styleImg = sizePreprocess(styleImg, opt.styleSize)

        local output = styleTransfer_mk(contentImg, styleImg)
        --local output = styleTransfer_wct(contentImg,styleImg)
        --local output = styleTransfer_adaIn(contentImg,styleImg)
        local savePath = paths.concat(opt.outputDir, contentName .. '_stylized_by_' .. styleName .. '_mk.jpg')
        print('Output image saved at: ' .. savePath)
        image.save(savePath, output)    
    end
end

