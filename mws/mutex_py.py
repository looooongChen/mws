import networkx as nx
import numpy as np
import math

class Mutex(object):

    def __init__(self, similarity='cos', lange_range=4):
        self.similarity = similarity
        self.lange_range = lange_range
    
    def _add_graph(self, image, mask=None):
        '''
        Args: 
            image: numpy array of size H x W x C or H x W x D x C, C is feature vector channel
        '''
        self.G = nx.Graph()
        sz = image.shape[:-1]
        mask = np.ones(sz) if mask is None else mask

        ind = np.nonzero(mask)
        if len(sz) == 2:
            self.G.add_nodes_from(ind[0] * sz[1] + ind[1])
        if len(sz) == 3:
            self.G.add_nodes_from(ind[0] * sz[1] * sz[2] + ind[1] * sz[2] + ind[2])

        l = self.lange_range
        if self.similarity == 'cos':
            image = image / (np.linalg.norm(image, ord=2, axis=-1, keepdims=True) + 1e-8)
            if len(sz) == 2:
                a0 = 3.1415/2 - np.arccos(np.clip(np.einsum('ijk,ijk->ij', image[:-1,:,:], image[1:,:,:]),0,1))
                a1 = 3.1415/2 - np.arccos(np.clip(np.einsum('ijk,ijk->ij', image[:,:-1,:], image[:,1:,:]),0,1)) 
                r0 = -1 * np.arccos(np.clip(np.einsum('ijk,ijk->ij', image[:-l,:,:], image[l:,:,:]),0,1)) + 0.25
                r1 = -1 * np.arccos(np.clip(np.einsum('ijk,ijk->ij', image[:,:-l,:], image[:,l:,:]),0,1)) + 0.25
            if len(sz) == 3:
                a0 = 3.1415/2 - np.arccos(np.clip(np.einsum('ijkm,ijkm->ijk', image[:-1,:,:,:], image[1:,:,:,:]),0,1))
                a1 = 3.1415/2 - np.arccos(np.clip(np.einsum('ijkm,ijkm->ijk', image[:,:-1,:,:], image[:,1:,:,:]),0,1))
                a2 = 3.1415/2 - np.arccos(np.clip(np.einsum('ijkm,ijkm->ijk', image[:,:,:-1,:], image[:,:,1:,:]),0,1)) 
                r0 = - np.arccos(np.clip(np.einsum('ijkm,ijkm->ijk', image[:-l,:,:,:], image[l:,:,:,:]),0,1)) + 0.25
                r1 = - np.arccos(np.clip(np.einsum('ijkm,ijkm->ijk', image[:,:-l,:,:], image[:,l:,:,:]),0,1)) + 0.25
                r2 = - np.arccos(np.clip(np.einsum('ijkm,ijkm->ijk', image[:,:,:-l,:], image[:,:,l:,:]),0,1)) + 0.25

        if len(sz) == 2:
            
            r, c = np.nonzero(np.logical_and(a0 > 0, mask[:-1,:]*mask[1:,:]>0))
            nodes = r * sz[1] + c
            neighbor = nodes + sz[1]
            for i, n in enumerate(nodes):
                self.G.add_edge(n, neighbor[i], weight=a0[r[i], c[i]])
            
            r, c = np.nonzero(np.logical_and(a1 > 0, mask[:,:-1]*mask[:,1:]>0))
            nodes = r * sz[1] + c
            neighbor = nodes + 1
            for i, n in enumerate(nodes):
                self.G.add_edge(n, neighbor[i], weight=a1[r[i], c[i]])

            r, c = np.nonzero(np.logical_and(r0 < 0, mask[:-l,:]*mask[l:,:]>0))
            nodes = r * sz[1] + c
            neighbor = nodes + sz[1] * l
            for i, n in enumerate(nodes):
                self.G.add_edge(n, neighbor[i], weight=r0[r[i], c[i]])
            
            r, c = np.nonzero(np.logical_and(r1 < 0, mask[:,:-l]*mask[:,l:]>0))
            nodes = r * sz[1] + c
            neighbor = nodes + l
            for i, n in enumerate(nodes):
                self.G.add_edge(n, neighbor[i], weight=r1[r[i], c[i]])
        
        if len(sz) == 3:
            
            r, c, d = np.nonzero(np.logical_and(a0 > 0, mask[:-1,:,:]*mask[1:,:,:]>0))
            nodes = r * sz[1] * sz[2] + c * sz[2] + d
            neighbor = nodes + sz[1] * sz[2]
            for i, n in enumerate(nodes):
                self.G.add_edge(n, neighbor[i], weight=a0[r[i], c[i], d[i]])
            
            r, c, d = np.nonzero(np.logical_and(a1 > 0, mask[:,:-1,:]*mask[:,1:,:]>0))
            nodes = r * sz[1] * sz[2] + c * sz[2] + d
            neighbor = nodes + sz[2]
            for i, n in enumerate(nodes):
                self.G.add_edge(n, neighbor[i], weight=a1[r[i], c[i], d[i]])

            r, c, d = np.nonzero(np.logical_and(a2 > 0, mask[:,:,:-1]*mask[:,:,1:]>0))
            nodes = r * sz[1] * sz[2] + c * sz[2] + d
            neighbor = nodes + 1
            for i, n in enumerate(nodes):
                self.G.add_edge(n, neighbor[i], weight=a2[r[i], c[i], d[i]])

            r, c, d = np.nonzero(np.logical_and(r0 < 0, mask[:-l,:,:]*mask[l:,:,:]>0))
            nodes = r * sz[1] * sz[2] + c * sz[2] + d
            neighbor = nodes + sz[1] * sz[2] * l
            for i, n in enumerate(nodes):
                self.G.add_edge(n, neighbor[i], weight=r0[r[i], c[i], d[i]])
            
            r, c, d = np.nonzero(np.logical_and(r1 < 0, mask[:,:-l,:]*mask[:,l:,:]>0))
            nodes = r * sz[1] * sz[2] + c * sz[2] + d
            neighbor = nodes + sz[2] * l
            for i, n in enumerate(nodes):
                self.G.add_edge(n, neighbor[i], weight=r1[r[i], c[i], d[i]])
            
            r, c, d = np.nonzero(np.logical_and(r2 < 0, mask[:,:,:-l]*mask[:,:,l:]>0))
            nodes = r * sz[1] * sz[2] + c * sz[2] + d
            neighbor = nodes + l
            for i, n in enumerate(nodes):
                self.G.add_edge(n, neighbor[i], weight=r2[r[i], c[i], d[i]])


    def clusters(self, image, mask=None, min_sz=10):

        self._add_graph(image, mask)

        sz = image.shape[:-1]
        label = np.ones(sz, dtype=np.uint32)

        G_attract = nx.Graph()
        G_attract.add_nodes_from(self.G)
        G_repulse = nx.Graph()
        G_repulse.add_nodes_from(self.G)
        for e in sorted(self.G.edges(data=True), key=lambda t: abs(t[2].get('weight')), reverse=True):
            if e[2].get('weight') > 0:
                if (not nx.has_path(G_attract, e[0], e[1])) and (not nx.has_path(G_repulse, e[0], e[1])):
                    G_attract.add_edge(e[0], e[1])
                    G_repulse.add_edge(e[0], e[1])
            else:
                if not nx.has_path(G_attract, e[0], e[1]):
                    G_repulse.add_edge(e[0], e[1])

        if len(sz) == 2:
            def index(ind):
                return np.floor_divide(ind, sz[1]), np.mod(ind, sz[1])
        if len(sz) == 3:
            def index(ind):
                return np.floor_divide(ind, sz[1]*sz[2]), np.floor_divide(np.mod(ind, sz[1]*sz[2]), sz[2]), np.mod(ind, sz[2])
        count = 0
        for c in nx.connected_components(G_attract):
            if len(c) > min_sz:
                count += 1
                ind = index(np.array(list(c)))
                label[ind] = count

        return label

if __name__ == '__main__': 
    import time
    import matplotlib.pyplot as plt

    # img = np.load('./emb.npy')[:,250:350,:100,:]
    embedding = np.load('./emb.npy')[:,:,:,:]

    # test 2d shape of shape 512 x 512 x (8)
    #   - without roi mask:  about 10s 
    #   - with roi mask: about 1s
    embedding = embedding[0]

    # test 3d shape of shape 9 x 512 x 512 x (8)
    #   - without roi mask:  about 170s 
    #   - with roi mask: about 65s
    # embedding = np.expand_dims(embedding[0], axis=0)
    # embedding = np.repeat(embedding, 9, axis=0)

    start = time.time()
    m = Mutex()
    # test with mask
    label = m.clusters(embedding, mask=embedding[...,0] != 0)
    # test no mask
    # label = m.clusters(embedding)
    print(time.time()-start)

    if len(embedding.shape) == 4:
        vis_emb = embedding[4,:,:,:3]
        vis_label = label[4]
    else:
        vis_emb = embedding[:,:,:3]
        vis_label = label

    plt.subplot(1,2,1)
    plt.imshow(vis_emb)
    plt.axis('off')
    plt.title('embedding')
    plt.subplot(1,2,2)
    plt.imshow(vis_label)
    plt.axis('off')
    plt.title('label')
    plt.show()



                



    
