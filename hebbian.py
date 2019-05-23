import io
import sys
import random
import numpy as np    
from PIL import Image


def image_to_individual_weight_matrix(path):
    imagea = open(path, 'rb').read()
    imga = Image.open(io.BytesIO(imagea))
    arra = np.asarray(imga)
    arra2 = arra>(127)
    arra3 = []
    for y in range(0,32):
        for x in range(0,32):
            if arra2[y][x][0] == True:
                arra3.append(1)
            else:
                arra3.append(-1)

    arra4 = np.array(arra3)[np.newaxis]
    arra6 = np.matmul(arra4.T,arra4)
    return arra6

def image_test(path,weight):
    imagea = open(path, 'rb').read()
    imga = Image.open(io.BytesIO(imagea))
    arra = np.asarray(imga)
    arra2 = arra>(127)
    arra3 = []
    for y in range(0,32):
        for x in range(0,32):
            if arra2[y][x][0] == True:
                arra3.append(1)
            else:
                arra3.append(-1)

    arra4 = np.array(arra3)[np.newaxis]
    arra6 = np.matmul(arra4,weight)
    arra7 = [1 if x>0 else 0 for x in arra6.T]
    return arra7

def rand_test(weight):
    arra3 = []
    for y in range(0,1024):
        x = np.random.randint(0,2)
        if x ==0:
            arra3.append(-1)
        else:
            arra3.append(1)
    arra4 = np.array(arra3)[np.newaxis]
    arra6 = np.matmul(arra4,weight)
    arra7 = [1 if x>0 else 0 for x in arra6.T]
    return arra7

a = image_to_individual_weight_matrix('train/a.png')
b = image_to_individual_weight_matrix('train/b.png')
c = image_to_individual_weight_matrix('train/c.png')
d = image_to_individual_weight_matrix('train/d.png')
e = image_to_individual_weight_matrix('train/e.png')


sum_of_weight = a+b+c

for x in range(1,1024):
    sum_of_weight[x][x] = 0

#at = rand_test(sum_of_weight)
test = image_test('test/tesc.png',sum_of_weight)

mat = np.reshape(test,(32,32))


img = Image.fromarray(np.uint8(mat * 255) , 'L')
img.show()



