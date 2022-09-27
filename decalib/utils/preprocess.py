import torch
import numpy as np
import cv2
import scipy
import scipy.io
import torchvision.transforms as transforms
import detectors

from torch.utils.data import Dataset, DataLoader
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob

def bbox2point(left, right, top, bottom, type='bbox'):
    ''' bbox from detector and landmarks are different
    '''
    if type=='kpt68':
        old_size = (right - left + bottom - top)/2*1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
    elif type=='bbox':
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
    else:
        raise NotImplementedError
    return old_size, center

def preProcessImage(image, bbox, bbox_type = 'kpt68', original = False):
    crop_size = 224
    scale = 1.25
    resolution_inp = crop_size
    face_detector = detectors.FAN()

    if len(image.shape) == 2:
        image = image[:,:,None].repeat(1,1,3)
    if len(image.shape) == 3 and image.shape[2] > 3:
        image = image[:,:,:3]
    
    h, w, _ = image.shape
    if original:
        left = bbox[0]; right=bbox[2]
        top = bbox[1]; bottom=bbox[3]
        src_pts = np.array([[0, 0], [0, right], [bottom, 0]])
    else: 
        bbox, bbox_type = face_detector.run(image)
        if len(bbox) < 4:
            print('no face detected! run original image')
            left = 0; right = h-1; top=0; bottom=w-1
        else:
            left = bbox[0]; right=bbox[2]
            top = bbox[1]; bottom=bbox[3]

        old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
        
        size = int(old_size*scale)
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])

    DST_PTS = np.array([[0,0], [0,resolution_inp - 1], [resolution_inp - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    image = image/255.

    dst_image = warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))
    dst_image = dst_image.transpose(2,0,1)
    return {'image': torch.tensor(dst_image).float(),
            'tform': torch.tensor(tform.params).float(),
            'original_image': torch.tensor(image.transpose(2,0,1)).float(),
            }