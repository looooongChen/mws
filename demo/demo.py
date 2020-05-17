import numpy as np
import math
import time
import matplotlib.pyplot as plt
from mws import *
from skimage.color import label2rgb

embedding = np.load('./emb.npy')[:,:,:,:]

#### test 2d ####

embedding = embedding[0]
m = MutexPixelEmbedding(similarity='cos', lange_range=4, min_size=10)
start = time.time()
seg = m.run(embedding, mask=embedding[:,:,0]!=0)
# seg = m.run(embedding)

print(time.time() - start)
plt.subplot(1,2,1)
plt.imshow(embedding[:,:,3])
plt.axis('off')
plt.title('pixel embedding')
plt.subplot(1,2,2)
plt.imshow(label2rgb(seg, bg_label=0))
plt.axis('off')
plt.title('segmentation')
plt.show()


#### test 3d ####

# embedding = np.expand_dims(embedding[0], axis=0)
# embedding = np.repeat(embedding, 50, axis=0)
# m = MutexPixelEmbedding(similarity='cos', lange_range=4, min_size=500)
# start = time.time()
# seg = m.run(embedding, mask=embedding[...,0]!=0)
# # seg = m.run(embedding)
# print(time.time() - start)

# plt.subplot(1,2,1)
# plt.imshow(embedding[0,:,:,0:3])
# plt.axis('off')
# plt.title('pixel embedding')
# plt.subplot(1,2,2)
# plt.imshow(label2rgb(seg[10], bg_label=0))
# plt.axis('off')
# plt.title('segmentation on slide 25')
# plt.show()



                



    
