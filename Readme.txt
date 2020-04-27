参照test.py
def synthesize(image,label,face_ind,triangles,angle,landmark3d,w,h)
    image[256,256,3]
    label[256,256,3]
    face_ind/triangles直接读入
    angle[pitch,yaw,roll]
    w,h = 256

label 为posmap格式
posmap生成参照PRnet_gen_posmap.py
把PRnet_gen_posmap.py放入PRnet项目下，生成posmap

Requirement:
- Python 2 or Python 3 

- Python packages:
  * numpy 
  * skimage (for reading&writing image)
  * scipy (for loading mat)
  * matplotlib (for show)
  * Cython (for compiling c++ file)


eample:

run test.py

example.png 为yaw轴旋转10度结果
