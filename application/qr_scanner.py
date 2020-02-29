# Standard library imports
import pdb
import math
import argparse
# Third party imports
import numpy as np
import cv2

# Local application imports
from functions import *
from argparse_fuctions import *

parser = argparse.ArgumentParser(description="Decodificacion QR")
group = parser.add_mutually_exclusive_group()
group.add_argument('--video', action= 'store_true', help = 'Select video mode')
group.add_argument('--image', action= 'store', help = 'Select image mode')
parser.add_argument('-s', action= 'store', help = 'Select source video by ID')
parser.add_argument('-p', action= 'store', help = 'Select path to image')
parser.add_argument('--debug', action= 'store_true', help = 'Select debug mode')
parser.add_argument('-b', action= 'store_true', help = 'Select binarization mode. 0 to OTSU - 1 THRES')

args = parser.parse_args()

if args.video:
    video_mode(int(args.s),int(args.debug),int(args.b))
    
if args.image:
    image_mode(str(args.image),int(args.debug), int(args.b))