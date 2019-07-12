require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'loadcaffe'

local cmd = torch.CmdLine()

cmd:option('-style_image', 'input/s035.jpg','Style image')
cmd:option('-content_image','input/c007.jpg','Content image')
cmd:option('-output_image','input/c007_s035_reshuffle.jpg','Output image')

cmd:option('-gpu', '0', 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-proto_file', 'models/vgg19_deploy.prototxt')
cmd:option('-model_file', 'models/vgg19.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-seed', -1)

cmd:option('-content_layers', 'relu4_2', 'layers for content')
cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for style')

local function main(params)

  local dtype, multigpu = setup_gpu(params)
  local loadcaffe_backend = params.backend
  local cnn = loadcaffe.load(params.proto_file, params.model_file, loadcaffe_backend):type(dtype)

  local content_image = image.load(params.content_image, 3)
  local content_image_caffe = preprocess(content_image):float()

  local style_image = image.load(params.style_image,3)
  local style_image_caffe = preprocess(style_image):float()

  local output_image = image.load(params.output_image,3)
  local output_image_caffe = preprocess(output_image):float()

  local content_layers = params.content_layers:split(",")
  local style_layers = params.style_layers:split(",")

  -- Set up the network, inserting style and content loss modules
  local content_losses, style_losses = {}, {}
  local next_content_idx, next_style_idx = 1, 1
  local net = nn.Sequential()
  
  for i = 1, #cnn do
    if next_content_idx <= #content_layers or next_style_idx <= #style_layers then
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
      if is_pooling and params.pooling == 'avg' then
        assert(layer.padW == 0 and layer.padH == 0)
        local kW, kH = layer.kW, layer.kH
        local dW, dH = layer.dW, layer.dH
        local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):type(dtype)
        local msg = 'Replacing max pooling at layer %d with average pooling'
        print(string.format(msg, i))
        net:add(avg_pool_layer)
      else
        net:add(layer)
      end
      if name == content_layers[next_content_idx] then
        print("Setting up content layer", i, ":", layer.name)
        
        local loss_module = nn.ContentLoss(1.0,false):type(dtype)
        
        net:add(loss_module)
        table.insert(content_losses, loss_module)
        next_content_idx = next_content_idx + 1
      end
      if name == style_layers[next_style_idx] then
        print("Setting up style layer  ", i, ":", layer.name)
        
        local loss_module = nn.StyleLoss(1.0, false):type(dtype)
        
        net:add(loss_module)
        table.insert(style_losses, loss_module)
        next_style_idx = next_style_idx + 1
      end
    end
  end
  net:type(dtype)
  print(net)

  -- Capture content targets
  for i = 1, #content_losses do
    content_losses[i].mode = 'capture'
  end
  print 'Capturing content targets'
  content_image_caffe = content_image_caffe:type(dtype)
  net:forward(content_image_caffe:type(dtype))

  -- Capture style targets
  for i = 1, #content_losses do
    content_losses[i].mode = 'none'
  end
  
  for j = 1, #style_losses do
    style_losses[j].mode = 'capture'
    style_losses[j].blend_weight = 1.0
  end
  net:forward(style_image_caffe:type(dtype))
  
  -- Set all loss modules to loss mode
  for i = 1, #content_losses do
    content_losses[i].mode = 'loss'
  end
  for i = 1, #style_losses do
    style_losses[i].mode = 'loss'
  end

  -- We don't need the base CNN anymore, so clean it up to save memory.
  cnn = nil
  for i=1, #net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' then
        module.gradWeight = nil
        module.gradBias = nil
    end
  end
  collectgarbage()

  -- Initialize the image
  if params.seed >= 0 then
    torch.manualSeed(params.seed)
  end

  -- forward and print losses
  output_image_caffe = output_image_caffe:type(dtype)
  local y = net:forward(output_image_caffe)

  for i, loss_module in ipairs(content_losses) do
    print(string.format('  Content %d loss: %f', i, loss_module.loss))
  end
  
  for i, loss_module in ipairs(style_losses) do
    print(string.format('  Style %d loss: %f', i, loss_module.loss))
  end
end

--/////
function setup_gpu(params)
  
  local multigpu = false
  params.gpu = tonumber(params.gpu) + 1
  local dtype = 'torch.FloatTensor'
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(params.gpu)
  dtype = 'torch.CudaTensor'
  params.backend = 'nn'
  return dtype, multigpu
end

--/////
function setup_multi_gpu(net, params)
  local DEFAULT_STRATEGIES = {
    [2] = {3},
  }
  local gpu_splits = nil
  if params.multigpu_strategy == '' then
    -- Use a default strategy
    gpu_splits = DEFAULT_STRATEGIES[#params.gpu]
    -- Offset the default strategy by one if we are using TV
    if params.tv_weight > 0 then
      for i = 1, #gpu_splits do gpu_splits[i] = gpu_splits[i] + 1 end
    end
  else
    -- Use the user-specified multigpu strategy
    gpu_splits = params.multigpu_strategy:split(',')
    for i = 1, #gpu_splits do
      gpu_splits[i] = tonumber(gpu_splits[i])
    end
  end
  assert(gpu_splits ~= nil, 'Must specify -multigpu_strategy')
  local gpus = params.gpu

  local cur_chunk = nn.Sequential()
  local chunks = {}
  for i = 1, #net do
    cur_chunk:add(net:get(i))
    if i == gpu_splits[1] then
      table.remove(gpu_splits, 1)
      table.insert(chunks, cur_chunk)
      cur_chunk = nn.Sequential()
    end
  end
  table.insert(chunks, cur_chunk)
  assert(#chunks == #gpus)

  local new_net = nn.Sequential()
  for i = 1, #chunks do
    local out_device = nil
    if i == #chunks then
      out_device = gpus[1]
    end
    new_net:add(nn.GPU(chunks[i], gpus[i], out_device))
  end

  return new_net
end

--/////////////////////
function build_filename(output_image, iteration)
  local ext = paths.extname(output_image)
  local basename = paths.basename(output_image, ext)
  local directory = paths.dirname(output_image)
  return string.format('%s/%s_%d.%s',directory, basename, iteration, ext)
end


-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end


-- Undo the above preprocessing.
function deprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(256.0)
  return img
end


-- Combine the Y channel of the generated image and the UV channels of the
-- content image to perform color-independent style transfer.
function original_colors(content, generated)
  local generated_y = image.rgb2yuv(generated)[{{1, 1}}]
  local content_uv = image.rgb2yuv(content)[{{2, 3}}]
  return image.yuv2rgb(torch.cat(generated_y, content_uv, 1))
end


-- Define an nn Module to compute content loss in-place
local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(strength, normalize)
  parent.__init(self)
  self.strength = strength
  self.target = torch.Tensor()
  self.normalize = normalize or false
  self.loss = 0
  self.crit = nn.MSECriterion()
  self.mode = 'none'
end

function ContentLoss:updateOutput(input)
  if self.mode == 'loss' then
    self.loss = self.crit:forward(input, self.target) * self.strength
    
    self.loss = self.loss/input:nElement()
  
  elseif self.mode == 'capture' then
    self.target:resizeAs(input):copy(input)
  end
  self.output = input
  return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
  if self.mode == 'loss' then
    if input:nElement() == self.target:nElement() then
      self.gradInput = self.crit:backward(input, self.target)
    end
    if self.normalize then
      self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
    end
    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
  else
    self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  end
  return self.gradInput
end


local Gram, parent = torch.class('nn.GramMatrix', 'nn.Module')

function Gram:__init()
  parent.__init(self)
end

function Gram:updateOutput(input)
  assert(input:dim() == 3)
  local C, H, W = input:size(1), input:size(2), input:size(3)
  local x_flat = input:view(C, H * W)
  self.output:resize(C, C)
  self.output:mm(x_flat, x_flat:t())
  return self.output
end

function Gram:updateGradInput(input, gradOutput)
  assert(input:dim() == 3 and input:size(1))
  local C, H, W = input:size(1), input:size(2), input:size(3)
  local x_flat = input:view(C, H * W)
  self.gradInput:resize(C, H * W):mm(gradOutput, x_flat)
  self.gradInput:addmm(gradOutput:t(), x_flat)
  self.gradInput = self.gradInput:view(C, H, W)
  return self.gradInput
end


-- Define an nn Module to compute style loss in-place
local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

function StyleLoss:__init(strength, normalize)
  parent.__init(self)
  self.normalize = normalize or false
  self.strength = strength
  self.target = torch.Tensor()
  self.mode = 'none'
  self.loss = 0

  self.gram = nn.GramMatrix()
  self.blend_weight = nil
  self.G = nil
  self.crit = nn.MSECriterion()
end

function StyleLoss:updateOutput(input)
  
  self.G = self.gram:forward(input)
  
  self.G:div(input:nElement()) -- 
  
  if self.mode == 'capture' then
    if self.blend_weight == nil then
      self.target:resizeAs(self.G):copy(self.G)
    elseif self.target:nElement() == 0 then
      self.target:resizeAs(self.G):copy(self.G):mul(self.blend_weight)
    else
      self.target:add(self.blend_weight, self.G)
    end
  elseif self.mode == 'loss' then
    
    self.loss = self.strength * self.crit:forward(self.G, self.target)


  end
  self.output = input
  return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
  if self.mode == 'loss' then
    local dG = self.crit:backward(self.G, self.target)
    dG:div(input:nElement())
    self.gradInput = self.gram:backward(input, dG)
    if self.normalize then
      self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
    end
    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
  else
    self.gradInput = gradOutput
  end
  return self.gradInput
end


local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(strength)
  parent.__init(self)
  self.strength = strength
  self.x_diff = torch.Tensor()
  self.y_diff = torch.Tensor()
end

function TVLoss:updateOutput(input)
  self.output = input
  return self.output
end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  local C, H, W = input:size(1), input:size(2), input:size(3)
  self.x_diff:resize(3, H - 1, W - 1)
  self.y_diff:resize(3, H - 1, W - 1)
  self.x_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.x_diff:add(-1, input[{{}, {1, -2}, {2, -1}}])
  self.y_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.y_diff:add(-1, input[{{}, {2, -1}, {1, -2}}])
  self.gradInput[{{}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
  self.gradInput[{{}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
  self.gradInput[{{}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end


local params = cmd:parse(arg)
main(params)
