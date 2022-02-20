import cv2, os, shutil
import numpy as np
from PIL import Image

# opening folders with data in them
flooding_true = "./data/FloodingTrue"
flooding_false = "./data/FloodingFalse"

sorted = []
used = []
filenames = []
for filename in os.listdir(flooding_true):

    temp = filename[0:4]
    if temp not in sorted:

        sorted.append(temp)
        filenames.append(filename)

sorted2 = []
used2 = []
filenames2 = []
for filename in os.listdir(flooding_false):

    temp = filename[0:4]
    if temp not in sorted2 and temp in sorted:

        sorted2.append(temp)
        filenames2.append(filename)

sorted = [i for i in sorted if i in sorted2]
#filenames = [i for i in filenames if i in filenames2]

used = []
final_filenames = []
for filename in filenames:

    temp_head = filename[0:4]
    temp_tail = filename[18:]

    for filename2 in filenames2:

        temp_head2 = filename2[0:4]
        temp_tail2 = filename2[19:]

        if temp_head == temp_head2 and temp_head not in used and temp_head2 not in used and filename:
            used.append(temp_head)
            final_filenames.append(filename)

print(sorted)
print(sorted2)
print('-------------------------------------------------------')
print(final_filenames)
print(filenames2)
print('-------------------------------------------------------')

for i in range(len(final_filenames)):

    shutil.copy(flooding_true + '/' + final_filenames[i],'./Image_Pairs/Pair' + str(i))
    shutil.copy(flooding_false + '/' + filenames2[i],'./Image_Pairs/Pair' + str(i))