### Faster RCNN Pytorch 多GPU 修改文档
#### 原生 Pytorch 接口
Pytorch 文档中提供的默认并行的接口

```python
data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None) 
 """Evaluates module(input) in parallel across the GPUs given in device_ids.

 This is the functional version of the DataParallel module.

 Args:
 module: the module to evaluate in parallel
 inputs: inputs to the module
 device_ids: GPU ids on which to replicate module
 output_device: GPU location of the output Use -1 to indicate the CPU.
 (default: device_ids[0])
 Returns:
 a Variable containing the result of module(input) located on
 output_device
 """
```

阅读源代码后发现其实内部调用了以下三个函数：

```python
replicas = self.replicate(self.module, self.device_ids[:len(inputs)]) # 将模型均分至N个device
outputs = self.parallel_apply(replicas, inputs, kwargs) # 运行前向
return self.gather(outputs, self.output_device)# 将所有的output搜集到一起
```

其中核心的 `paraller_apply` 的代码为：

```python 
def parallel_apply(modules, inputs, kwargs_tup=None):
    assert len(modules) == len(inputs)
    if kwargs_tup:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    # Fast track
    if len(modules) == 1:
        return (modules[0](*inputs[0], **kwargs_tup[0]))
    lock = threading.Lock()
    results = {}

    def _worker(i, module, input, kwargs, results, lock):
        var_input = input
        while not isinstance(var_input, Variable):
            var_input = var_input[0]
        try:
            with torch.cuda.device_of(var_input): # 根据var_input 设定前向的GPU
                output = module(*input, **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    threads = [threading.Thread(target=_worker,
                                args=(i, module, input, kwargs, results, lock),
                                )
               for i, (module, input, kwargs) in
               enumerate(zip(modules, inputs, kwargs_tup))]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs
```
算法核心包括三个步骤是，处理输入参数，启动多个线程处理，并将算好的结果进行归并
#### Faster RCNN 代码中修改
* 修改 `paraller_apply.py` 中 `paraller_apply` 函数
原版的代码 `paraller_apply` 会根据input选择前向函数的GPU id。Faster RCNN pytorch代码中输入的参数都是cpu中来的，需要修改为从模型中获取相关的GPU id：

```python 
for param in module.parameters():         
   deviceid = param.get_device()
   try:
   		with torch.cuda.device(deviceid):
      # with torch.cuda.device_of(var_input):
      		output = module(*input, **kwargs)
```

* 修改调用方式
原版的代码输入为多个参数，而`paraller_apply` 接受的输入只有一个，是一个对应参数的list 因此修改调用的代码为：

```python
	for eachstep :
	    device_ids =[0,1] # 设置GPU 
	    output_device = [0] # 设置最终结果的位置
	
	    replicas = nn.parallel.replicate(net, device_ids) # 模型set到两个GPU
	    blobs_dict =[]
	    inputs = []        
	    
	    for i in range(0,nGPU):    
	        blobs = data_layer.forward()
	        input = []
	        input.append(blobs['im_info'])
	        input.append(blobs['gt_boxes'])
	        input.append(blobs['gt_ishard'])
	        input.append(blobs['dontcare_areas'])
	        inputs.append(input) 
	    print "prepare data finish"
	    replicas = replicas[:len(inputs)]
	    outputs = nn.parallel.parallel_apply(replicas,inputs) # 送入input 跑修改后的前
	    out = nn.parallel.gather(outputs, output_device)
```

* 修改 `faster_rcnn` 内部代码
内部有多处设计显存分配的地方需要注意其中包括：

	* 修正 `faster_rcnn/network.py` 中 `np_to_variable` 函数中默认开在`GPU id ==0`问题：
	
	```python
		def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
 	   		v = Variable(torch.from_numpy(x).type(dtype))
 	   		if is_cuda:    	
   			#  v = v.cuda()   #原先未制定分配的GPU位置，统一分配在GPU 0 上面
				deviceid = torch.cuda.current_device() # 修改为寻找当前使用的GPU id 
				v = v.cuda(deviceid) #分配显存
    		return v
	```
	
	* 当CPU 计算后，当前的GPU id 会自动恢复到i的0 ,因此修改于CPU混合计算的layer，包括：

		*	RPN `proposal_layer` 函数：
		
		```python				
		def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchor_scales):
		    	deviceid = rpn_cls_prob_reshape.get_device() # 获取输入变量GPU id
		    	rpn_cls_prob_reshape = rpn_cls_prob_reshape.data.cpu().numpy()
		    	rpn_bbox_pred = rpn_bbox_pred.data.cpu().numpy()
		    	x = proposal_layer_py(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchor_scales)
		    	with torch.cuda.device(deviceid): # 指定分配时候的GPU id
		        	x = network.np_to_variable(x, is_cuda=True)
		    	return x.view(-1, 5)
		```
		
		* RPN `anchor_target_layer` 函数:
		
		```python 
		   def anchor_target_layer(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride, anchor_scales):
		       """
		       rpn_cls_score: for pytorch (1, Ax2, H, W) bg/fg scores of previous conv layer
		       gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
		       gt_ishard: (G, 1), 1 or 0 indicates difficult or not
		       dontcare_areas: (D, 4), some areas may contains small objs but no labelling. D may be 0
		       im_info: a list of [image_height, image_width, scale_ratios]
		       _feat_stride: the downsampling ratio of feature map to the original input image
		       anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
		       ----------
		       Returns
		       ----------
		       rpn_labels : (1, 1, HxA, W), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
		       rpn_bbox_targets: (1, 4xA, H, W), distances of the anchors to the gt_boxes(may contains some transform)
		                       that are the regression objectives
		       rpn_bbox_inside_weights: (1, 4xA, H, W) weights of each boxes, mainly accepts hyper param in cfg
		       rpn_bbox_outside_weights: (1, 4xA, H, W) used to balance the fg/bg,
		       beacuse the numbers of bgs and fgs mays significiantly different
		       """
			   deviceid = rpn_cls_score.get_device() # 获取输入变量GPU id
		       
			   rpn_cls_score = rpn_cls_score.data.cpu().numpy()
		       rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
		           anchor_target_layer_py(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride, anchor_scales)
			   with torch.cuda.device(deviceid):    # 指定分配时候的GPU id
		           rpn_labels = network.np_to_variable(rpn_labels, is_cuda=True, dtype=torch.LongTensor)
		           rpn_bbox_targets = network.np_to_variable(rpn_bbox_targets, is_cuda=True)
		           rpn_bbox_inside_weights = network.np_to_variable(rpn_bbox_inside_weights, is_cuda=True)
		           rpn_bbox_outside_weights = network.np_to_variable(rpn_bbox_outside_weights, is_cuda=True)
		
		       return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
		```

		* Faster_RCNN `proposal_target_layer` 函数：
		
		```python
		  def proposal_target_layer(rpn_rois, gt_boxes, gt_ishard, dontcare_areas, num_classes):
		        """
		        ----------
		        rpn_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
		        gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
		        # gt_ishard: (G, 1) {0 | 1} 1 indicates hard
		        dontcare_areas: (D, 4) [ x1, y1, x2, y2]
		        num_classes
		        ----------
		        Returns
		        ----------
		        rois: (1 x H x W x A, 5) [0, x1, y1, x2, y2]
		        labels: (1 x H x W x A, 1) {0,1,...,_num_classes-1}
		        bbox_targets: (1 x H x W x A, K x4) [dx1, dy1, dx2, dy2]
		        bbox_inside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
		        bbox_outside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
		        """
		    
		        deviceid = rpn_rois.get_device() # 获取输入变量GPU id
		        rpn_rois = rpn_rois.data.cpu().numpy()
		       
		        rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
		            proposal_target_layer_py(rpn_rois, gt_boxes, gt_ishard, dontcare_areas, num_classes)
		        
		        # print labels.shape, bbox_targets.shape, bbox_inside_weights.shape
		        with torch.cuda.device(deviceid): # 指定分配时候的GPU id
		            rois = network.np_to_variable(rois, is_cuda=True)
		            logging.info('"deviceid {0}"'.format(deviceid))
		            labels = network.np_to_variable(labels, is_cuda=True, dtype=torch.LongTensor)
		            bbox_targets = network.np_to_variable(bbox_targets, is_cuda=True)
		            bbox_inside_weights = network.np_to_variable(bbox_inside_weights, is_cuda=True)
		            bbox_outside_weights = network.np_to_variable(bbox_outside_weights, is_cuda=True)
		
		        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
			```

	* 修改build loss 函数
		在RPN 和 Faster_RCNN 的 build loss 函数中有分配内存的部分，修改成分配到制定的GPU 上。
		* RPN `build loss` 函数：
  	
	  	```python
	  	def build_loss(self, rpn_cls_score_reshape, rpn_bbox_pred, rpn_data):
	        # classification loss
	        rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2)
	        rpn_label = rpn_data[0].view(-1)
			# 修改	
	        deviceid = rpn_cls_score.get_device() # 获取 GPU id
	        rpn_keep = Variable(rpn_label.data.ne(-1).nonzero().squeeze()).cuda(deviceid) # 指定 GPU id
	        
			rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
	    
	 		# 修改	
	        deviceid = rpn_label.get_device() # 获取 GPU id    
	        rpn_keep = Variable(rpn_label.data.ne(-1).nonzero().squeeze()).cuda(deviceid) # 指定 GPU id
	
		    rpn_label = torch.index_select(rpn_label, 0, rpn_keep)
	
	        fg_cnt = torch.sum(rpn_label.data.ne(0))
	
	        rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)
	
	        # box loss
	        rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
	        rpn_bbox_targets = torch.mul(rpn_bbox_targets, rpn_bbox_inside_weights)
	        rpn_bbox_pred = torch.mul(rpn_bbox_pred, rpn_bbox_inside_weights)
	
	        rpn_loss_box = F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, size_average=False) / (fg_cnt + 1e-4)
	        return rpn_cross_entropy, rpn_loss_box
		```
		* Faster RCNN `build loss` 函数：
	
	   ```python 
	    def build_loss(self, cls_score, bbox_pred, roi_data):
	        # classification loss
	        label = roi_data[1].squeeze()
	        fg_cnt = torch.sum(label.data.ne(0))
	        bg_cnt = label.data.numel() - fg_cnt
	
	        # for log
	        if self.debug:
	            maxv, predict = cls_score.data.max(1)
				self.tp = torch.sum(predict[:fg_cnt].eq(label.data[:fg_cnt])) if fg_cnt > 0 else 0
				self.tf = torch.sum(predict[fg_cnt:].eq(label.data[fg_cnt:])) if bg_cnt > 0 and len(predict)-fg_cnt > 0 else 0
	            self.fg_cnt = fg_cnt
	            self.bg_cnt = bg_cnt
	
	        ce_weights = torch.ones(cls_score.size()[1])
	        ce_weights[0] = float(fg_cnt) / (bg_cnt+1e-4)
			
			# 修改
			deviceid = cls_score.get_device() # 获取 GPU id 
	        ce_weights = ce_weights.cuda(deviceid) # 指定 GPU id
	        
			cross_entropy = F.cross_entropy(cls_score, label, weight=ce_weights)
	
	        # bounding box regression L1 loss
	        bbox_targets, bbox_inside_weights, bbox_outside_weights = roi_data[2:]
	        bbox_targets = torch.mul(bbox_targets, bbox_inside_weights)
	        bbox_pred = torch.mul(bbox_pred, bbox_inside_weights)
	
	        loss_box = F.smooth_l1_loss(bbox_pred, bbox_targets, size_average=False) / (fg_cnt + 1e-4)
	
	        return cross_entropy, loss_box
			```
			
	* ROI Pooling 中有分配内存的动作，默认也是分配在GPU 0 上，需要修改`fater_rcnn/ROI_pooling/functions/roi_pool.py` 中`forward` 函数:

	```python
	def forward(self, features, rois):
	    batch_size, num_channels, data_height, data_width = features.size()
	    num_rois = rois.size()[0]
	    output = torch.zeros(num_rois, num_channels, self.pooled_height, self.pooled_width)
	    argmax = torch.IntTensor(num_rois, num_channels, self.pooled_height, self.pooled_width).zero_()
	
	    if not features.is_cuda:
	        _features = features.permute(0, 2, 3, 1)
	        roi_pooling.roi_pooling_forward(self.pooled_height, self.pooled_width, self.spatial_scale,
	 _features, rois, output)
	        # output = output.cuda()
	 else:
	        # output = output.cuda() # 修改前
	        # argmax = argmax.cuda() # 修改前
					
	        output = output.cuda(features.get_device()) # 修改为分配到指定位置
			argmax= argmax.cuda(argmax.get_device()) # 修改为分配到指定位置
	
	        roi_pooling.roi_pooling_forward_cuda(self.pooled_height, self.pooled_width, self.spatial_scale,
	 features, rois, output, argmax)
	        self.output = output
	        self.argmax = argmax
	        self.rois = rois
	        self.feature_size = features.size()
	    return output
	```
	
	* 为了返回最终的`loss`值，需要修改`Faster_rcnn` 中 `forward` 函数
	
	```python
	   def forward(self, im_data, im_info, gt_boxes=None, gt_ishard=None, dontcare_areas=None):
	        
	
	        res4_features, rois = self.rpn(im_data, im_info, gt_boxes, gt_ishard, dontcare_areas)
	
	        if self.training:
	            roi_data = self.proposal_target_layer(rois, gt_boxes, gt_ishard, dontcare_areas, self.n_classes)
	            rois = roi_data[0]
	
	
	        # roi pool
	        pooled_features = self.roi_pool(res4_features, rois)
	    
	
	        # print 'pooled_features',pooled_features.size()
	        res5_features = self.res5_features(pooled_features)
	
	        logging.info("res5 feature finished")
	        x = res5_features.view(res5_features.size()[0], -1)
	
	        cls_score = self.score_fc(x)
	        cls_prob = F.softmax(cls_score)
	        bbox_pred = self.bbox_fc(x)
	
	        if self.training:
	            self.cross_entropy, self.loss_box = self.build_loss(cls_score, bbox_pred, roi_data)
	
	        #return cls_prob, bbox_pred, rois 
	
	        return self.cross_entropy + self.loss_box * 10 # 修改为返回loss
	```

此外镜像中pytorch代码不是最新版，`parallel_apply` 不是最新的版本，会有错误，需要修改成最新版的pytouch
至此，多GPU的fasterRCNN 就可以正常运行了

#### 仍有的问题：

多GPU返回的loss 虽然可以用 `nn.parallel.gather` 收集到一块GPU上，但是收集到loss 并不能自动调用backward来计算,会报 `some of weight/gradient/input tensors are located on different GPUs`. 错误，需要进一步的修正
