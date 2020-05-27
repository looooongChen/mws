# *mws* - Mutex Watershed 

This is an implementation of mutex watershed, for 2d and 3d image segmentation without seeds.

[The Mutex Watershed: Efficient, Parameter-Free Image Partitioning](http://openaccess.thecvf.com/content_ECCV_2018/html/Steffen_Wolf_The_Mutex_Watershed_ECCV_2018_paper.html)

## Implementations:

### mutex watershed parameters
- 2D mutex watershed: 
    - short range attractive 4 neighbor (along 2 axes)
    - lang range repulsive 8 neighbor (along 2 axes and 2 diagonals) 
- 3D mutex watershed: 
    - short range attractive 6 neighbor (along 3 axes)
    - lang range repulsive 18 neighbor (along 3 axes and 6 diagnals in the coordinate planes)

### implementation details
- connectivity query: union find + weighted tree + path compression
- sort: quick sort + consideration of the case, where large amount of same values

### running speed
The running time is reported on a Dell XPS13 Laptop: 
- Intel(R) Core(TM) i7-8550 CPU @ 1.8 GHz 2.00 GHz
- 16 GB memory
- Windows 10 

Running time:
- 2D image 512 x 512: ~ 0.37s
- 3D image 10 x 512 x 512: ~ 7.7s
- 3D image 50 x 512 x 512 : ~ 41.8s 
- 3D image 100 x 512 x 512: ~ 123.8s

Applying masks can reduce the computation time by reducing the edges involved. Althought the actual effect depends on the mask, remeber to use it if available.

Runing time with mask:
- 2D image 512 x 512 with mask: ~ 0.15s
- 3D image 10 x 512 x 512 with mask: ~ 2.8s
- 3D image 50 x 512 x 512 with mask: ~ 15.5s 
- 3D image 100 x 512 x 512 with mask: ~ 33.23s

## Installation

```python
# clone the git repository
git clone https://github.com/looooongChen/mws.git
cd mws
# install
python setup.py install 
# or if you only want to complie the C extension
python setup.py build
```

## Usage

The usage is simple by just calling: 
```python
mws2d(attractive, repulsive, repulsive_range, min_size)
'''
Args: 
    attractive: (H x W, 2) attractive weights with neigbor pixels (with distance 1) along directions
        - (dim1)-axis
        - (dim2)-axis
    repulsive: (H x H x 4) repulsive weights with long range pixels along directions
        - (dim1)-axis, 
        - (dim2)-axis, 
        - (dim1,dim2)-diagnal, 
        - (dim1,-dim2)-diagnal direction
    rep_range: distance of the long range repulsive weights
    min_size: the minimal number of pixels to form a cluster, default=0
Return: (H x W) segmentation map
'''

mws3d(attractive, repulsive, repulsive_range, min_sz)
'''
Args: 
    attractive: (H x W x D, 3) attractive weights with neigbor pixels (with distance 1) along directions
        - (dim1)-axis
        - (dim2)-axis
        - (dim3)-axis
    repulsive: (H x W x D x 9) repulsive weights with long range pixels along directions
        - (dim1)-axis
        - (dim2)-axis
        - (dim3)-axis
        - (dim1,dim2)-diagnal
        - (dim1,-dim2)-diagnal
        - (dim1,dim3)-diagnal
        - (dim1,-dim3)-diagnal
        - (dim2,dim3)-diagnal
        - (dim2,-dim3)-diagnal
    rep_range: distance of the long range repulsive weights
    min_size: the minimal number of pixels to form a cluster, default=0
Return: (H x W x D) segmentation map
'''
```

Note: 
- only attractive / repulisve != 0 will be processed, to reduce the computation workload, set non-concerning edge to zero
- margin area, where no neighbor pixel exists, will be ingored, for example, the last pixel of a row has no right-side neighbor pixel 

Combined with deep learning approaches, the attractive and repulsive weights can be explicitly predicted<sup>\[1\]</sup> or implicitly estimated through pixel embedding<sup>\[2\]</sup>.

If you work on pixel embedding, the following class may be also helpful:

```python
class MutexCosinePixelEmbedding(object):
    ...

class MutexEuclideanPixelEmbedding(object):
    ...
```

## Reference

\[1\] [Superhuman Accuracy on the Snemi3d Connectomics Challenge](https://arxiv.org/abs/1706.00120)  

\[2\] [Instance Segmentation of Biomedical Images with an Object-aware Embedding Learned with Local Constraints](https://www.researchgate.net/publication/340826279_Instance_Segmentation_of_Biomedical_Images_with_an_Object-aware_Embedding_Learned_with_Local_Constraints)
\[3\] [Semantic Instance Segmentation with a Discriminative Loss Function](https://arxiv.org/abs/1708.02551)
