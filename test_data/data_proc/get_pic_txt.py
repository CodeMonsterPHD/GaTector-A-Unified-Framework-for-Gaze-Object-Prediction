import os
images_path='/data1/jinyang/GaTector/data_proc/images1'
image_list=sorted(os.listdir(images_path))
with open('/data1/jinyang/GaTector/data_proc/test.txt','w+')as f:
    for i in range(len(image_list)):
        f.write(str(i)+'\n')
print('Done!')