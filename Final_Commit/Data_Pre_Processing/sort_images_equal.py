import cv2, os, shutil
import numpy as np
from PIL import Image

# opening folders with data in them
flooding_true = "./data/FloodingTrue"
flooding_false = "./data/FloodingFalse"

paired = []
file_pairs = {}
for filename in os.listdir(flooding_true):

    temp = filename[0:4]
    if temp not in paired:

        flag = 0
        for filename2 in os.listdir(flooding_true):

            if filename2 != filename and filename2[0:4] == temp and flag == 0:
                flag = 1
                paired.append(temp)
                file_pairs[filename] = filename2

paired2 = []
file_pairs2 = {}
for filename in os.listdir(flooding_false):

    temp = filename[0:4]
    if temp not in paired2:

        flag = 0
        for filename2 in os.listdir(flooding_false):

            if filename2 != filename and filename2[0:4] == temp and flag == 0:
                flag = 1
                paired2.append(temp)
                file_pairs2[filename] = filename2

print(paired)
print(file_pairs)
print('------------------------')
print(paired2)
print(file_pairs2)

count = 0
for i in file_pairs:

    os.mkdir('./Image_Pairs_Equal/Pair' + str(count))
    shutil.copy(flooding_true + '/' + i,'./Image_Pairs_Equal/Pair' + str(count))
    shutil.copy(flooding_true + '/' + file_pairs[i],'./Image_Pairs_Equal/Pair' + str(count))
    count += 1

for e in file_pairs2:

    os.mkdir('./Image_Pairs_Equal/Pair' + str(count))
    shutil.copy(flooding_false + '/' + e,'./Image_Pairs_Equal/Pair' + str(count))
    shutil.copy(flooding_false + '/' + file_pairs2[e],'./Image_Pairs_Equal/Pair' + str(count))
    count += 1
