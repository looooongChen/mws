import numpy as np
from mws.src.mws import mws2d_c, mws3d_c

def mws2d(attractive, repulsive, repulsive_range, min_size=None):
    '''
    python arguments: 
       attractive (H x W x 2), positive float32:
           attractive weights along (dim1)-axis, (dim2)-axis direction
       repulsive (H x W, 4), positive float32:
           repulsive weights along (dim1)-axis, (dim2)-axis, (dim1,dim2)-diagnal, (dim1,-dim2)-diagnal direction
           margin area will be ingored
       rep_range l: distance of repulsive force
       min_size: minimal number of pixels to form a cluster
    
    Note: 
       - only attractive / repulisve != 0 will be processed, to reduce the computation workload, set non-concerning edge to zeor
       - margin area, where no neighbor pixel exists, will be ingored, for example, the last pixel of a row has no right-side neighbor pixel  
    '''
    assert attractive.shape[:-1] == repulsive.shape[:-1]
    assert attractive.shape[-1] == 2
    assert repulsive.shape[-1] == 4
    assert repulsive_range > 0
    min_size = 0 if min_size is None else min_size
    return mws2d_c(attractive.astype(np.float32), np.abs(repulsive.astype(np.float32)), int(repulsive_range), int(min_size))

def mws3d(attractive, repulsive, repulsive_range, min_size):
    '''
    python arguments: 
       attractive (H x W x D x 3), positive float32:
           attractive weights along (dim1)-axis, (dim2)-axis, (dim3)-axis direction
       repulsive (H x W x D x 9), positive float32:
           repulsive weights along (dim1)-axis, (dim2)-axis, (dim3)-axis (dim1,dim2)-diagnal, (dim1,-dim2)-diagnal, (dim1,dim3)-diagnal, (dim1,-dim3)-diagnal, (dim2,dim3)-diagnal, (dim2,-dim3)-diagnal direction
           margin area will be ingored
       rep_range l: distance of repulsive force
       min_size: minimal number of pixels to form a cluster
    
    Note: 
       - only attractive / repulisve != 0 will be processed, to reduce the computation workload, set non-concerning edge to zeor
       - margin area, where no neighbor pixel exists, will be ingored, for example, the last pixel of a row has no right-side neighbor pixel
    '''
    assert attractive.shape[:-1] == repulsive.shape[:-1]
    assert attractive.shape[-1] == 3
    assert repulsive.shape[-1] == 9
    assert repulsive_range > 0
    min_size = 0 if min_size is None else min_size
    return mws3d_c(attractive.astype(np.float32), np.abs(repulsive.astype(np.float32)), int(repulsive_range), int(min_size))
    

class MutexCosinePixelEmbedding(object):

    def __init__(self, lange_range=4, min_size=0):
        self.lange_range = lange_range
        self.min_size = min_size
    
    def run(self, image, mask=None):
        '''
        Args: 
            image: numpy array of size H x W x C or H x W x D x C, C is feature vector channel
        '''
        sz = image.shape[:-1]
        mask = np.ones((*sz, 1)) if mask is None else np.expand_dims(mask, axis=-1)

        l = self.lange_range
        
        if len(sz) == 2:
            image = image / (np.linalg.norm(image, ord=2, axis=-1, keepdims=True) + 1e-8)
            # attractive along axis
            a0 = np.einsum('ijk,ijk->ij', image[:-1,:,:], image[1:,:,:])
            a1 = np.einsum('ijk,ijk->ij', image[:,:-1,:], image[:,1:,:])
            # repulsive along axis
            r0 = np.einsum('ijk,ijk->ij', image[:-l,:,:], image[l:,:,:])
            r1 = np.einsum('ijk,ijk->ij', image[:,:-l,:], image[:,l:,:])
            # repulsive along diagnal
            r01 = np.einsum('ijk,ijk->ij', image[:-l,:-l,:], image[l:,l:,:])
            r01_ = np.einsum('ijk,ijk->ij', image[:-l,l:,:], image[l:,:-l,:])
            
            # pad array
            a0 = np.pad(a0,((0,1),(0,0)), 'constant', constant_values=0)
            a1 = np.pad(a1,((0,0),(0,1)), 'constant', constant_values=0)
            r0 = np.pad(r0,((0,l),(0,0)), 'constant', constant_values=0)
            r1 = np.pad(r1,((0,0),(0,l)), 'constant', constant_values=0)
            r01 = np.pad(r01,((0,l),(0,l)), 'constant', constant_values=0)
            r01_ = np.pad(r01_,((0,l),(l,0)), 'constant', constant_values=0) 
            # get the attr and repulisive weights
            attr = 3.1415/2 - np.arccos(np.clip(np.stack([a0, a1], axis=-1), 0, 1))
            attr = attr * mask
            rep = -1 * np.arccos(np.clip(np.stack([r0, r1, r01, r01_], axis=-1), 0, 1))+0.2
            rep = rep * mask

        if len(sz) == 3:
            # attractive along axis
            a0 = np.einsum('ijkm,ijkm->ijk', image[:-1,:,:,:], image[1:,:,:,:])
            a1 = np.einsum('ijkm,ijkm->ijk', image[:,:-1,:,:], image[:,1:,:,:])
            a2 = np.einsum('ijkm,ijkm->ijk', image[:,:,:-1,:], image[:,:,1:,:])
            # repulsive along axis
            r0 = np.einsum('ijkm,ijkm->ijk', image[:-l,:,:,:], image[l:,:,:,:])
            r1 = np.einsum('ijkm,ijkm->ijk', image[:,:-l,:,:], image[:,l:,:,:])
            r2 = np.einsum('ijkm,ijkm->ijk', image[:,:,:-l,:], image[:,:,l:,:])
            # repulsive along diagnal on the coordinate planes
            r01 = np.einsum('ijkm,ijkm->ijk', image[:-l,:-l,:,:], image[l:,l:,:,:])
            r01_ = np.einsum('ijkm,ijkm->ijk', image[:-l,l:,:,:], image[l:,:-l,:,:])
            r02 = np.einsum('ijkm,ijkm->ijk', image[:-l,:,:-l,:], image[l:,:,l:,:])
            r02_ = np.einsum('ijkm,ijkm->ijk', image[:-l,:,l:,:], image[l:,:,:-l,:])
            r12 = np.einsum('ijkm,ijkm->ijk', image[:,:-l,:-l,:], image[:,l:,l:,:])
            r12_ = np.einsum('ijkm,ijkm->ijk', image[:,:-l,l:,:], image[:,l:,:-l,:])
           
            # pad array
            a0 = np.pad(a0,((0,1),(0,0),(0,0)), 'constant', constant_values=0)
            a1 = np.pad(a1,((0,0),(0,1),(0,0)), 'constant', constant_values=0)
            a2 = np.pad(a2,((0,0),(0,0),(0,1)), 'constant', constant_values=0)
            r0 = np.pad(r0,((0,l),(0,0),(0,0)), 'constant', constant_values=0)
            r1 = np.pad(r1,((0,0),(0,l),(0,0)), 'constant', constant_values=0)
            r2 = np.pad(r2,((0,0),(0,0),(0,l)), 'constant', constant_values=0)
            r01 = np.pad(r01,((0,l),(0,l),(0,0)), 'constant', constant_values=0)
            r01_ = np.pad(r01_,((0,l),(l,0),(0,0)), 'constant', constant_values=0)
            r02 = np.pad(r02,((0,l),(0,0),(0,l)), 'constant', constant_values=0)
            r02_ = np.pad(r02_,((0,l),(0,0),(l,0)), 'constant', constant_values=0)
            r12 = np.pad(r12,((0,0),(0,l),(0,l)), 'constant', constant_values=0)
            r12_ = np.pad(r12_,((0,0),(0,l),(l,0)), 'constant', constant_values=0)
            # get the attr and repulisive weights
            attr = 3.1415/2 - np.arccos(np.clip(np.stack([a0,a1,a2], axis=-1), 0, 1))
            attr = attr * mask
            rep = -1 * np.arccos(np.clip(np.stack([r0,r1,r2,r01,r01_,r02,r02_,r12,r12_], axis=-1), 0, 1))+0.25
            rep = rep * mask

        if len(sz) == 2:
            return mws2d(attr, rep, l, self.min_size) * np.squeeze(mask)

        if len(sz) == 3:
            return mws3d(attr, rep, l, self.min_size) * np.squeeze(mask)


class MutexEuclideanPixelEmbedding(object):

    def __init__(self, lange_range=4, min_size=0, cluster_distance=3):
        self.lange_range = lange_range
        self.min_size = min_size
        self.cluster_distance = cluster_distance # average distance between clusters
    
    def run(self, image, mask=None):
        '''
        Args: 
            image: numpy array of size H x W x C or H x W x D x C, C is feature vector channel
        '''
        sz = image.shape[:-1]
        mask = np.ones((*sz, 1)) if mask is None else np.expand_dims(mask, axis=-1)

        l = self.lange_range
        
        if len(sz) == 2:
            image = image / (np.linalg.norm(image, ord=2, axis=-1, keepdims=True) + 1e-8)
            # attractive along axis
            a0 = np.sqrt(np.sum((image[:-1,:,:]-image[1:,:,:])**2, axis=-1))
            a1 = np.sqrt(np.sum((image[:,:-1,:]-image[:,1:,:])**2, axis=-1))
            # repulsive along axis
            r0 = np.sqrt(np.sum((image[:-l,:,:]-image[l:,:,:])**2, axis=-1))
            r1 = np.sqrt(np.sum((image[:,:-l,:]-image[:,l:,:])**2, axis=-1))
            # repulsive along diagnal
            r01 = np.sqrt(np.sum((image[:-l,:-l,:]-image[l:,l:,:])**2, axis=-1))
            r01_ = np.sqrt(np.sum((image[:-l,l:,:]-image[l:,:-l,:])**2, axis=-1))
            
            # pad array
            a0 = np.pad(a0,((0,1),(0,0)), 'constant', constant_values=0)
            a1 = np.pad(a1,((0,0),(0,1)), 'constant', constant_values=0)
            r0 = np.pad(r0,((0,l),(0,0)), 'constant', constant_values=0)
            r1 = np.pad(r1,((0,0),(0,l)), 'constant', constant_values=0)
            r01 = np.pad(r01,((0,l),(0,l)), 'constant', constant_values=0)
            r01_ = np.pad(r01_,((0,l),(l,0)), 'constant', constant_values=0) 
            # get the attr and repulisive weights
            attr = self.cluster_distance - np.stack([a0, a1], axis=-1)
            attr = attr * mask
            rep = np.stack([r0, r1, r01, r01_], axis=-1)
            rep = rep * mask

        if len(sz) == 3:
            # attractive along axis
            a0 = np.sqrt(np.sum((image[:-1,:,:,:]-image[1:,:,:,:])**2, axis=-1))
            a1 = np.sqrt(np.sum((image[:,:-1,:,:]-image[:,1:,:,:])**2, axis=-1))
            a2 = np.sqrt(np.sum((image[:,:,:-1,:]-image[:,:,1:,:])**2, axis=-1))
            # repulsive along axis
            r0 = np.sqrt(np.sum((image[:-l,:,:,:]-image[l:,:,:,:])**2, axis=-1))
            r1 = np.sqrt(np.sum((image[:,:-l,:,:]-image[:,l:,:,:])**2, axis=-1))
            r2 = np.sqrt(np.sum((image[:,:,:-l,:]-image[:,:,l:,:])**2, axis=-1))
            # repulsive along diagnal on the coordinate planes
            r01 = np.sqrt(np.sum((image[:-l,:-l,:,:]-image[l:,l:,:,:])**2, axis=-1))
            r01_ = np.sqrt(np.sum((image[:-l,l:,:,:]-image[l:,:-l,:,:])**2, axis=-1))
            r02 = np.sqrt(np.sum((image[:-l,:,:-l,:]-image[l:,:,l:,:])**2, axis=-1))
            r02_ = np.sqrt(np.sum((image[:-l,:,l:,:]-image[l:,:,:-l,:])**2, axis=-1))
            r12 = np.sqrt(np.sum((image[:,:-l,:-l,:]-image[:,l:,l:,:])**2, axis=-1))
            r12_ = np.sqrt(np.sum((image[:,:-l,l:,:]-image[:,l:,:-l,:])**2, axis=-1))
           
            # pad array
            a0 = np.pad(a0,((0,1),(0,0),(0,0)), 'constant', constant_values=0)
            a1 = np.pad(a1,((0,0),(0,1),(0,0)), 'constant', constant_values=0)
            a2 = np.pad(a2,((0,0),(0,0),(0,1)), 'constant', constant_values=0)
            r0 = np.pad(r0,((0,l),(0,0),(0,0)), 'constant', constant_values=0)
            r1 = np.pad(r1,((0,0),(0,l),(0,0)), 'constant', constant_values=0)
            r2 = np.pad(r2,((0,0),(0,0),(0,l)), 'constant', constant_values=0)
            r01 = np.pad(r01,((0,l),(0,l),(0,0)), 'constant', constant_values=0)
            r01_ = np.pad(r01_,((0,l),(l,0),(0,0)), 'constant', constant_values=0)
            r02 = np.pad(r02,((0,l),(0,0),(0,l)), 'constant', constant_values=0)
            r02_ = np.pad(r02_,((0,l),(0,0),(l,0)), 'constant', constant_values=0)
            r12 = np.pad(r12,((0,0),(0,l),(0,l)), 'constant', constant_values=0)
            r12_ = np.pad(r12_,((0,0),(0,l),(l,0)), 'constant', constant_values=0)
            # get the attr and repulisive weights
            attr = self.cluster_distance - np.stack([a0,a1,a2], axis=-1)
            attr = attr * mask
            rep = np.stack([r0,r1,r2,r01,r01_,r02,r02_,r12,r12_], axis=-1)
            rep = rep * mask

        if len(sz) == 2:
            return mws2d(attr, rep, l, self.min_size) * np.squeeze(mask)

        if len(sz) == 3:
            return mws3d(attr, rep, l, self.min_size) * np.squeeze(mask)


        



                



    
