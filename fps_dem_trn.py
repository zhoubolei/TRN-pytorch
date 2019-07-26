
# Author: Dinesh Palanisamy
# Website: dineshp.ai
# Demo: https://youtu.be/6PAvFzV4Yfo





# FPS and webcam vido stream from Adrian Rosebrock
# (http://www.pyimagesearch.com)
# USAGE
# BE SURE TO INSTALL 'imutils' PRIOR TO EXECUTING THIS COMMAND

# import the necessary packages
from __future__ import print_function
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import cv2
import pandas as pd


def putIterationsPerSec(frame, label):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    cv2.putText(frame, "class label: " + label,
                (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame


# created a *threaded *video stream, allow the camera senor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from webcam...")
vs = VideoStream(src=0).start()
fps = FPS().start()

import os
import re
import cv2
import argparse
import functools
import subprocess
import numpy as np
from PIL import Image
import moviepy.editor as mpy

import torchvision
import torch.nn.parallel
import torch.optim
from torch.autograd import Variable
from models import TSN
import transforms
from torch.nn import functional as F


def load_frames(frames, num_frames=8):
    if len(frames) >= num_frames:
        return frames[::int(np.ceil(len(frames) / float(num_frames)))]
    else:
        raise (ValueError('Video must have at least {} frames'.format(num_frames)))


def render_frames(frames, prediction):
    rendered_frames = []
    for frame in frames:
        img = np.array(frame)
        height, width, _ = img.shape
        cv2.putText(img, prediction,
                    (1, int(height / 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        rendered_frames.append(img)
    return rendered_frames


parser = argparse.ArgumentParser(description="test TRN on a single video")
# group = parser.add_mutually_exclusive_group(required=True)
parser.add_argument('--video_file', type=str, default='')
parser.add_argument('--frame_folder', type=str, default='')
parser.add_argument('--modality', type=str, default='RGB',
                    choices=['RGB', 'Flow', 'RGBDiff'], )
parser.add_argument('--dataset', type=str, default='jester',
                    choices=['something', 'jester', 'moments', 'somethingv2'])
parser.add_argument('--rendered_output', type=str, default='test')
parser.add_argument('--arch', type=str, default="InceptionV3")
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--test_segments', type=int, default=8)
parser.add_argument('--img_feature_dim', type=int, default=256)
parser.add_argument('--consensus_type', type=str, default='TRNmultiscale')
parser.add_argument('--weights', type=str,
                    default='pretrain/TRN_jester_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar')

args = parser.parse_args()

# Get dataset categories.
categories_file = 'pretrain/{}_categories.txt'.format(args.dataset)
categories = [line.rstrip() for line in open(categories_file, 'r').readlines()]
num_class = len(categories)

args.arch = 'InceptionV3' if args.dataset == 'moments' else 'BNInception'

# Load model.
net = TSN(num_class,
          args.test_segments,
          args.modality,
          base_model=args.arch,
          consensus_type=args.consensus_type,
          img_feature_dim=args.img_feature_dim, print_spec=False)

checkpoint = torch.load(args.weights)
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)
net.cuda().eval()

# Initialize frame transforms.
transform = torchvision.transforms.Compose([
    transforms.GroupScale(net.scale_size),
    transforms.GroupCenterCrop(net.input_size),
    transforms.Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
    transforms.ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
    transforms.GroupNormalize(net.input_mean, net.input_std),
])

pred = "";
bufferf = []

# loop over some frames...this time using the threaded stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = putIterationsPerSec(frame, pred)

    # frame = cv2.resize(frame, (640, 480))

    crop_img = frame[0:720, 0:720]
    # h,w = crop_img.shape[:2]
    input_img = crop_img;
    input_pill = Image.fromarray(input_img)
    cv2.imshow("Frame", crop_img)
    key = cv2.waitKey(1) & 0xFF

    if (len(bufferf) < 16):
        bufferf.append(input_pill)
    else:
        input_frames = load_frames(bufferf)
        print(input_frames)
        data = transform(input_frames)
        input = Variable(data.view(-1, 3, data.size(1), data.size(2)).unsqueeze(0).cuda(), volatile=True)

        # with torch.no_grad():
        logits = net(input)
        h_x = torch.mean(F.softmax(logits, 1), dim=0).data
        probs, idx = h_x.sort(0, True)
        pred = categories[idx[0]]
        bufferf[:-1] = bufferf[1:];
        bufferf[-1] = input_pill
        # check to see if the frame should be displayed to our screen

    # update the FPS counter
    fps.update()
