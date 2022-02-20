import cv2, os
import numpy as np
from PIL import Image

string1 = 'FLOODING: True'
string2 = 'FLOODING: False'

# opening a text file
file = open("./dev_set/Image_Set (1)/timeseries.txt", "r")

# setting flag and index to 0
flag1 = 0
flag2 = 0

index1 = 0
index2 = 0

line1 = ''
line2 = ''

# Loop through the file line by line
for line in file:

    if flag1 == 0:
        index1 += 1
        line1 = line

    if flag2 == 0:
        index2 += 1
        line2 = line

    if string1 in line:
        flag1 = 1

    if string2 in line:
        flag2 = 1

    if flag1 == 1 and flag2 == 1:
        break

print('String', string1, 'Found In Line', index1)
print('String', string2, 'Found In Line', index2)
print('-----------------------------')
print(line1)
print(line2)

base_path = "./dev_set/Image_Set (6)/B01_series.tif"
new_path = "./test_output"

'''
read = cv2.imread(base_path)
outfile = 'test.jpg'
cv2.imwrite(new_path+outfile,read,[int(cv2.IMWRITE_JPEG_QUALITY), 200])
'''

def read_tiff(path):
    """
    path - Path to the multipage-tiff file
    """
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))

    temp = np.array(images)

    final = Image.fromarray(temp, 'RGB')
    final.save('test.png')
    final.show()

read_tiff(base_path)

# closing text file
file.close()