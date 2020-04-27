from augmentation import synthesize
import cv2
import numpy as np

image = cv2.imread('./10173-other_3-1.jpg')
face_ind = np.loadtxt('./Data/uv-data/face_ind.txt').astype(np.int32)
triangles = np.loadtxt('./Data/uv-data/triangles.txt').astype(np.int32)
#input_image = image/255.
mark3d = []
landmark = open('./landmark.txt')
for line in landmark.readlines():
    lines = line.strip('\n')
    word = lines.split(' ')
    mark3d.append(round(float(word[0])))

# mark2d = []

pos_gt = np.load('./10173-other_3-1.npy')
#pos_gt = np.array(pos_gt).astype(np.float32)
[h, w, _] = image.shape
img,pos,vtx,mark2d = synthesize(image,pos_gt,face_ind,triangles,[0,10,0],mark3d,h,w)
# for i in range(len(mark3d)):
#     x = vtx[mark3d[i]][0]
#     y = vtx[mark3d[i]][1]
#     mark2d.append([x,y])
for i in range(len(mark2d)):
    center = list(mark2d[i])
    #print(center)
    cv2.circle(img,(int(center[0]),int(center[1])),2,(100,100,100),2)
cv2.imshow('img',img)
cv2.waitKey(0)