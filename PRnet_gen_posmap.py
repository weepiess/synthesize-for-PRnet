import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from time import time
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2 as cv
from api import PRN
from utils.write import write_obj_with_colors
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
def RotateClockWise90(img):
    trans_img = cv.transpose( img )
    new_img = cv.flip(trans_img, 1)
    return new_img

# ---- init PRN
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU number, -1 for CPU
prn = PRN(is_dlib = True) 


# ------------- load data
image_folder = '/media/weepies/Seagate Backup Plus Drive/3DMM/3d-pixel/evaluate_data/image/'
#image_folder = '/home/weepies/3DMM/3DFAW/output/'
save_folder = '/media/weepies/Seagate Backup Plus Drive/3DMM/3dp_chosse/3dpixel/finetune_rec/'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

types = ('*.jpg', '*.png')
image_path_list= []
for files in types:
    image_path_list.extend(glob(os.path.join(image_folder, files)))
total_num = len(image_path_list)

for i, image_path in enumerate(image_path_list):
    print('step: ',i)
    # read image
    print(image_path)
    
    name = image_path.split('.jpg')[0].split('image/')[1]
    print('name: ',name)
    if name == '10022-15-1':
        image = imread(image_path)
        pts_l = []
        kpt_f = open('/media/weepies/Seagate Backup Plus Drive/3DMM/3d-pixel/evaluate_data/pts/'+name+'.pts')
        for lines in kpt_f.readlines():
            line = lines.strip('\n')
            word = line.split(' ')
            if len(word) == 2:
                pts_l.append([round(float(word[0])),round(float(word[1]))])
        kpt_f.close()
        for i in range(len(pts_l)):
            y = 720 - pts_l[i][1]
            x = pts_l[i][0] 
            pts_l[i][0] = 720 - y
            pts_l[i][1] = x
        kpt1 = pts_l[0:33:2]
        #print('1: ',len(kpt1))
        kpt2 = pts_l[33:64]
        #print('2: ',len(kpt2))
        kpt3 = pts_l[84:104]
        #print('3: ',len(kpt3))
        ptso = []
        for i in range(len(kpt1)):
            ptso.append(kpt1[i])
        for i in range(len(kpt2)):
            ptso.append(kpt2[i])
        for i in range(len(kpt3)):
            ptso.append(kpt3[i])
        pts_n = np.array(ptso)

        # cv.imshow('ori',image)
        # cv.waitKey(0)
        img_ori = cv.flip(image,0)
        image = RotateClockWise90(img_ori)

        pos = prn.process(image,pts_n) # use dlib to detect face
        # for i in range(len(pts_n)):
        #     center = list(pts_n[i])
        #     #print(center)
        #     cv.circle(image,(int(center[0]),int(center[1])),2,(100,100,100),2)
        # cv.imshow('pic',image)
        # cv.waitKey(0)
        # cv.imshow('pic',picture)
        # cv.waitKey(0)
        # -- Basic Applications
        # get landmarks
        kpt = prn.get_landmarks(pos)

        # for i in range(len(pts_n)):
        #     center = list(pts_n[i])
        #     #print(center)
        #     cv.circle(image,(int(center[0]),int(center[1])),2,(100,100,100),2)
        # cv.imshow('pic',image)
        # cv.waitKey(0)
        # for i in range(len(kpt)):
        #     kpt[i,0] = format(float(kpt[i,0]),'.4f')
        #     kpt[i,1] = format(float(kpt[i,1]),'.4f')
        #     kpt[i,2] = format(float(kpt[i,2]),'.4f')
        #     #print('kpt:',kpt[i,1])
        # # 3D vertices
        vertices = prn.get_vertices(pos)
        for i in range(len(vertices)):
            vertices[i,0] = format((vertices[i,0]),'.6f')
            vertices[i,1] = format((vertices[i,1]),'.6f')
            vertices[i,2] = format((vertices[i,2]),'.6f')
            #print('v0: ',vertices[i,0])
        # corresponding colors
        colors = prn.get_colors(image, vertices)

        # -- save
        name = image_path.strip().split('/')[-1][:-4]
        np.savetxt(os.path.join(save_folder, name + '.txt'), kpt,fmt='%.6f') 
        write_obj_with_colors(os.path.join(save_folder, 'mesh'+name + '.obj'), vertices, prn.triangles, colors) #save 3d face(can open with meshlab)

        #sio.savemat(os.path.join(save_folder, 'mesh'+name + '.mat'), {'vertices': vertices, 'colors': colors, 'triangles': prn.triangles})

