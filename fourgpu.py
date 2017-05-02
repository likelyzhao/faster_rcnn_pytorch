import os
import torch
import numpy as np
from datetime import datetime
from faster_rcnn import network
#from faster_rcnn.faster_rcnn import FasterRCNN, RPN
from faster_rcnn.faster_rcnn_inceptionresnetv2_imgsize1000 import FasterRCNN, RPN
from faster_rcnn.utils.timer import Timer

import faster_rcnn.roi_data_layer.roidb as rdl_roidb
from faster_rcnn.roi_data_layer.layer import RoIDataLayer
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file

from  torch.autograd import Variable
import torch.nn as nn

try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


imdb_name = 'imagenet_2015_train'
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
pretrained_model = '/disk2/data/pytorch_models/inceptionresnetv2-d579a627.pth'
output_dir = '/disk2/data/pytorch_models/trained_models/inceptionresnetv2_imgsize1000/saved_model3'

start_step = 0
end_step = 4000000
save_model_steps = 100000
lr_decay_steps = {2400000, 3200000}
lr_decay = 1. / 10

rand_seed = 1024
_DEBUG = True
use_tensorboard = True
remove_all_log = False   # remove all historical experiments in TensorBoard
exp_name = None  # the previous experiment name in TensorBoard

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
print "load cfg from file"
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

# load data
print "load data"
imdb = get_imdb(imdb_name)
print "prere roidb"
rdl_roidb.prepare_roidb(imdb)
print "done"
roidb = imdb.roidb
print "ROIDataLayer"
data_layer = RoIDataLayer(roidb, imdb.num_classes)

# load netZZ
print "initialize faster rcnn"

nGPU = torch.cuda.device_count()
print torch.cuda.device_count()

nets = []

# for  i in range(1,nGPU):
net = FasterRCNN(classes=imdb.classes, debug=_DEBUG)
network.weights_normal_init(net, dev=0.01)
#network.load_pretrained_npy(net, pretrained_model)
network.load_pretrained_pth(net, pretrained_model)
print "add net"
net.cuda()
print ("set in device {}",)
net.train()

# tensorboad
use_tensorboard = use_tensorboard and CrayonClient is not None
if use_tensorboard:
    cc = CrayonClient(hostname='127.0.0.1')
    if remove_all_log:
        cc.remove_all_experiments()
    if exp_name is None:
        exp_name = datetime.now().strftime('vgg16_%m-%d_%H-%M')
        exp = cc.create_experiment(exp_name)
    else:
        exp = cc.open_experiment(exp_name)

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()
print "start training"
for step in range(start_step, end_step + 1):
    print "step:{}".format(step)
    # get one batch

    device_ids =[0,1]
    output_device = [0]

    replicas = nn.parallel.replicate(net, device_ids)
    print len(replicas)
#    inputs = nn.parallel.scatter(input, device_ids)
    blobs_dict =[]
    inputs = []	    
    imdatas=[]
	
    for i in range(0,nGPU):    
    	blobs = data_layer.forward()
	input = []
	input.append(blobs["data"])
        blobs_dict.append(blobs)
     	input.append(blobs['im_info'])
   	input.append(blobs['gt_boxes'])
    	input.append(blobs['gt_ishard'])
    	input.append(blobs['dontcare_areas'])
  	inputs.append(input) 
    print "prepare data finish"

#    out = net(blobs)
    
 #   print (out)
 #   inputs = nn.parallel.scatter(blobs_dict, device_ids)
    
    replicas = replicas[:len(inputs)]
 #   for param in replicas[0].parameters():
    	#print (param.get_device())
    outputs = nn.parallel.parallel_apply(replicas,inputs)

    out = nn.parallel.gather(outputs,0)

    out = out[0]+out[1]
    print (out.backward())



      # forward
    net(im_data, im_info, gt_boxes, gt_ishard, dontcare_areas)
    loss = net.loss + net.rpn.loss

    if _DEBUG:
        tp += float(net.tp)
        tf += float(net.tf)
        fg += net.fg_cnt
        bg += net.bg_cnt

    train_loss += loss.data[0]
    step_cnt += 1

    # backward
    optimizer.zero_grad()
    loss.backward()
    network.clip_gradient(net, 10.)
    optimizer.step()

    if step % disp_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration

        log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch)' % (
            step, blobs['im_name'], train_loss / step_cnt, fps, 1. / fps)
        log_print(log_text, color='green', attrs=['bold'])

        if _DEBUG:
            log_print('\tTP: %.2f%%, TF: %.2f%%, fg/bg=(%d/%d)' % (tp / (fg + 1e-4)
                                                                   * 100., tf / (bg + 1e-4) * 100., fg / step_cnt, bg / step_cnt))
            log_print('\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f' % (
                net.rpn.cross_entropy.data.cpu().numpy(
                )[0], net.rpn.loss_box.data.cpu().numpy()[0],
                net.cross_entropy.data.cpu().numpy()[0], net.loss_box.data.cpu().numpy()[0])
            )
        re_cnt = True

    if use_tensorboard and step % log_interval == 0:
        exp.add_scalar_value('train_loss', train_loss / step_cnt, step=step)
        exp.add_scalar_value('learning_rate', lr, step=step)
        if _DEBUG:
            exp.add_scalar_value('true_positive', tp / fg * 100., step=step)
            exp.add_scalar_value('true_negative', tf / bg * 100., step=step)
            losses = {'rpn_cls': float(net.rpn.cross_entropy.data.cpu().numpy()[0]),
                      'rpn_box': float(net.rpn.loss_box.data.cpu().numpy()[0]),
                      'rcnn_cls': float(net.cross_entropy.data.cpu().numpy()[0]),
                      'rcnn_box': float(net.loss_box.data.cpu().numpy()[0])}
            exp.add_scalar_dict(losses, step=step)

    if (step % save_model_steps == 0) and step > 0:
        save_name = os.path.join(output_dir, 'faster_rcnn_{}.h5'.format(step))
        network.save_net(save_name, net)
        print('save model: {}'.format(save_name))
    if step in lr_decay_steps:
        lr *= lr_decay
        print "learning rate decay:{}".format(lr)
        optimizer = torch.optim.SGD(
            params[528:], lr=lr, momentum=momentum, weight_decay=weight_decay)

    if re_cnt:
        tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False
print 

