#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <iostream>
#include <numpy/arrayobject.h>

class UF {

    private:
        npy_intp node_num = 0;
        npy_intp* nodes;
        npy_intp* cp_size;

    public:
        UF(npy_intp size) {
            node_num = size;
            nodes = new npy_intp[size];
            cp_size = new npy_intp[size];
            for (npy_intp i = 0; i < size; i++) {
                nodes[i] = i;
                cp_size[i] = 1;
            }
        }

        npy_intp root(npy_intp n) {
            while (n != nodes[n]) {
                nodes[n] = nodes[nodes[n]];
                n = nodes[n];
            }
            return n;
        }

        npy_intp cluster_size(npy_intp n) {
            return cp_size[n];
        }

        bool is_connected(npy_intp n1, npy_intp n2) {
            return root(n1) == root(n2);
        }

        void connect(npy_intp n1, npy_intp n2) {
            npy_intp root1 = root(n1);
            npy_intp root2 = root(n2);
            if (root1 != root2) {
                if (cp_size[root1] < cp_size[root2]) {
                    nodes[root1] = root2;
                    cp_size[root2] += cp_size[root1];
                }
                else {
                    nodes[root2] = root1;
                    cp_size[root1] += cp_size[root2];
                }
            }
        }

        void flatten_root_tree(void) {
            for (npy_intp i = 0; i < node_num; i++) {
                nodes[i] = root(i);
            }
        }
};

void swap_index(npy_intp* a, npy_intp* b) {
    npy_intp t = *a;
    *a = *b;
    *b = t;
}

void swap_value(float* a, float* b) {
    float t = *a;
    *a = *b;
    *b = t;
}

npy_intp partition(float* arr, npy_intp* index, npy_intp low, npy_intp high){
    float pivot = arr[high];  // pivot 
    npy_intp i = low - 1;  // Index of smaller element
    npy_intp count = 0;

    for (npy_intp j = low; j <= high - 1; j++) {
        if (arr[j] == pivot) {
            // std::cout << "count" << count << std::endl;
            count++;
        }
        else{
            count = 0;
        }
        if (arr[j] <= pivot) {
            i++;
            swap_value(&arr[i], &arr[j]);
            swap_index(&index[i], &index[j]);
        }
    }
    swap_value(&arr[i + 1], &arr[high]);
    swap_index(&index[i + 1], &index[high]);
    return (i + 1 - count/2);
}

void quickSort(float* arr, npy_intp* index, npy_intp low, npy_intp high) {
    // std::cout << low << "   " << high << std::endl;
    if (low < high) {
        npy_intp pi = partition(arr, index, low, high);

        quickSort(arr, index, low, pi - 1);
        quickSort(arr, index, pi + 1, high);
    }
}

static PyObject* mws2d_c (PyObject *self, PyObject *args) {

    // python arguments: 
    //    attractive (H x W x 2), positive float32:
    //        attractive weights along (dim1)-axis, (dim2)-axis direction
    //    repulsive (H x W, 4), positive float32:
    //        repulsive weights along (dim1)-axis, (dim2)-axis, (dim1,dim2)-diagnal, (dim1,-dim2)-diagnal direction
    //        margin area will be ingored
    //    rep_range l, int:
    //    min_sz, it: minimal number of pixels to form a cluster
    
    // Note: 
    //    - only attractive / repulisve != 0 will be processed, to reduce the computation workload, set non-concerning edge to zeor
    //    - margin area, where no neighbor pixel exists, will be ingored, for example, the last pixel of a row has no right-side neighbor pixel  

    PyArrayObject *attractive = NULL, *repulsive = NULL, *res = NULL;
    int rep_range;
    int min_sz;
 
    if (!PyArg_ParseTuple(args, "O!O!ii", &PyArray_Type, &attractive, &PyArray_Type, &repulsive, &rep_range, &min_sz))
        return NULL;

    npy_intp *dims = PyArray_DIMS(attractive);
    // npy_intp *dims = PyArray_DIMS(repulsive);

    // set impossible edge connections to 0
    for (npy_intp i=0; i<dims[0]; i++) {
        *(float *)PyArray_GETPTR3(attractive, i, dims[1]-1, 1) = 0;
        for(npy_intp d=0; d<rep_range; d++){
            *(float *)PyArray_GETPTR3(repulsive, i, dims[1]-d-1, 1) = 0;
            *(float *)PyArray_GETPTR3(repulsive, i, dims[1]-d-1, 2) = 0;
            *(float *)PyArray_GETPTR3(repulsive, i, d, 3) = 0;
        }
    }
    for (npy_intp j=0; j<dims[1]; j++) {
        *(float *)PyArray_GETPTR3(attractive, dims[0]-1, j, 0) = 0;
        for(npy_intp d=0; d<rep_range; d++){
            *(float *)PyArray_GETPTR3(repulsive, dims[0]-d-1, j, 0) = 0;
            *(float *)PyArray_GETPTR3(repulsive, dims[0]-d-1, j, 2) = 0;
            *(float *)PyArray_GETPTR3(repulsive, dims[0]-d-1, j, 3) = 0;
        }
    }

    // debug
    // res = (PyArrayObject*)PyArray_SimpleNew(3,dims,NPY_FLOAT);
    // memcpy((float*)PyArray_DATA(res), (float*)PyArray_DATA(repulsive), dims[0]*dims[1]*dims[2]*sizeof(float));
    // return PyArray_Return(res);

    UF G_attr(dims[0]*dims[1]);
    UF G_rep(dims[0]*dims[1]);

    npy_intp dims_res[2];
    dims_res[0] = dims[0];
    dims_res[1] = dims[1];

    res = (PyArrayObject*)PyArray_SimpleNew(2,dims_res,NPY_INT32);

    // sort edges (in increasing order)
    float* data_attr = (float*)PyArray_DATA(attractive);
    npy_intp* index_attr = (npy_intp*)malloc(sizeof(npy_intp) * dims[0] * dims[1] * 2);
    for (npy_intp i = 0; i < dims[0] * dims[1] * 2; i++) {
        index_attr[i] = i;
    }
    quickSort(data_attr, index_attr, 0, dims[0]*dims[1] * 2-1);

    float* data_rep = (float*)PyArray_DATA(repulsive);
    npy_intp* index_rep = (npy_intp*)malloc(sizeof(npy_intp) * dims[0] * dims[1] * 4);
    for (npy_intp i = 0; i < dims[0] * dims[1] * 4; i++) {
        index_rep[i] = i;
    }
    quickSort(data_rep, index_rep, 0, dims[0]*dims[1]*4 - 1);

    // 
    npy_intp p_attr = dims[0] * dims[1] * 2 - 1;
    npy_intp p_rep = dims[0] * dims[1] * 4 - 1;
    npy_intp i, j, d, n1, n2;
    while(1){
        if (data_attr[p_attr] < 0.0001){
            p_attr = 0;
        }
        if (data_rep[p_rep] < 0.0001){
            p_rep = 0;
        }
        if (p_attr == 0 && p_rep == 0){
            break;
        }
        // process an attract edge 
        if( data_attr[p_attr] > data_rep[p_rep] ){
            i = index_attr[p_attr] / (dims[1] * 2);
            j = (index_attr[p_attr] % (dims[1] * 2)) / 2;
            d = index_attr[p_attr] % 2;
            n1 = i * dims[1] + j;
            n2 = (d == 0) ? n1+dims[1] : n1+1;
            if (!G_attr.is_connected(n1, n2) && !G_rep.is_connected(n1, n2)){
                G_attr.connect(n1, n2);
                G_rep.connect(n1, n2);
            }
            p_attr--;
        }
        // process a repulsive edge
        else{
            i = index_rep[p_rep] / (dims[1] * 4);
            j = (index_rep[p_rep] % (dims[1] * 4)) / 4;
            d = index_rep[p_rep] % 4;
            n1 = i * dims[1] + j;
            switch(d % 4) {
                case 0: n2 = n1 + dims[1]; break;
                case 1: n2 = n1 + 1; break;
                case 2: n2 = n1 + dims[1] + 1; break;
                default: n2 = n1 + dims[1] - 1; break;
            }
            if(!G_attr.is_connected(n1, n2)){
                G_rep.connect(n1, n2);
            }
            p_rep--;
        }
    }
    
    npy_intp root, size;
    for(int ind=0; ind<dims[0]*dims[1]; ind++){
        i = ind / dims[1];
        j = ind % dims[1];
        root = G_attr.root(ind);
        size = G_attr.cluster_size(root);
        if(size > min_sz){
            *(npy_intp *)PyArray_GETPTR2(res, i, j) = root + 1;
        }
        else{
            *(npy_intp *)PyArray_GETPTR2(res, i, j) = 0;
        }
        
    }

    return PyArray_Return(res);
}

static PyObject* mws3d_c (PyObject *self, PyObject *args) {

    // python arguments: 
    //    attractive (H x W x D x 3), positive float32:
    //        attractive weights along (dim1)-axis, (dim2)-axis, (dim3)-axis direction
    //    repulsive (H x W x D 9), positive float32:
    //        repulsive weights along (dim1)-axis, (dim2)-axis, (dim3)-axis (dim1,dim2)-diagnal, (dim1,-dim2)-diagnal, (dim1,dim3)-diagnal, (dim1,-dim3)-diagnal, (dim2,dim3)-diagnal, (dim2,-dim3)-diagnal direction
    //        margin area will be ingored
    //    rep_range l, int:
    //    min_sz, it: minimal number of pixels to form a cluster
    
    // Note: 
    //    - only attractive / repulisve != 0 will be processed, to reduce the computation workload, set non-concerning edge to zeor
    //    - margin area, where no neighbor pixel exists, will be ingored, for example, the last pixel of a row has no right-side neighbor pixel  

    PyArrayObject *attractive = NULL, *repulsive = NULL, *res = NULL;
    int rep_range;
    int min_sz;
 
    if (!PyArg_ParseTuple(args, "O!O!ii", &PyArray_Type, &attractive, &PyArray_Type, &repulsive, &rep_range, &min_sz))
        return NULL;

    npy_intp *dims = PyArray_DIMS(attractive);
    // npy_intp *dims = PyArray_DIMS(repulsive);

    // set impossible edge connections to 0
    for (npy_intp i=0; i<dims[0]; i++) {
        for (npy_intp j=0; j<dims[1]; j++){
            *(float *)PyArray_GETPTR4(attractive, i, j, dims[2]-1, 2) = 0;
            for(npy_intp d=0; d<rep_range; d++){
                *(float *)PyArray_GETPTR4(repulsive, i, j, dims[2]-d-1, 2) = 0;
                *(float *)PyArray_GETPTR4(repulsive, i, j, dims[2]-d-1, 5) = 0;
                *(float *)PyArray_GETPTR4(repulsive, i, j, d, 6) = 0;
                *(float *)PyArray_GETPTR4(repulsive, i, j, dims[2]-d-1, 7) = 0;
                *(float *)PyArray_GETPTR4(repulsive, i, j, d, 8) = 0;
            }
        }
    }
    for (npy_intp i=0; i<dims[0]; i++) {
        for (npy_intp k=0; k<dims[2]; k++){
            *(float *)PyArray_GETPTR4(attractive, i, dims[1]-1, k, 1) = 0;
            for(npy_intp d=0; d<rep_range; d++){
                *(float *)PyArray_GETPTR4(repulsive, i, dims[1]-d-1, k, 1) = 0;
                *(float *)PyArray_GETPTR4(repulsive, i, dims[1]-d-1, k, 3) = 0;
                *(float *)PyArray_GETPTR4(repulsive, i, d, k, 4) = 0;
                *(float *)PyArray_GETPTR4(repulsive, i, dims[1]-d-1, k, 7) = 0;
                *(float *)PyArray_GETPTR4(repulsive, i, dims[1]-d-1, k, 8) = 0;
            }
        }
    }
    for (npy_intp j=0; j<dims[1]; j++) {
        for (npy_intp k=0; k<dims[2]; k++){
            *(float *)PyArray_GETPTR4(attractive, dims[0]-1, j, k, 0) = 0;
            for(npy_intp d=0; d<rep_range; d++){
                *(float *)PyArray_GETPTR4(repulsive, dims[0]-d-1, j, k, 0) = 0;
                *(float *)PyArray_GETPTR4(repulsive, dims[0]-d-1, j, k, 3) = 0;
                *(float *)PyArray_GETPTR4(repulsive, dims[0]-d-1, j, k, 4) = 0;
                *(float *)PyArray_GETPTR4(repulsive, dims[0]-d-1, j, k, 5) = 0;
                *(float *)PyArray_GETPTR4(repulsive, dims[0]-d-1, j, k, 6) = 0;
            }
        }
    }

    // debug
    // res = (PyArrayObject*)PyArray_SimpleNew(4,dims,NPY_FLOAT);
    // memcpy((float*)PyArray_DATA(res), (float*)PyArray_DATA(repulsive), dims[0]*dims[1]*dims[2]*dims[3]*sizeof(float));
    // return PyArray_Return(res);

    UF G_attr(dims[0]*dims[1]*dims[2]);
    UF G_rep(dims[0]*dims[1]*dims[2]);

    npy_intp dims_res[3];
    dims_res[0] = dims[0];
    dims_res[1] = dims[1];
    dims_res[2] = dims[2];

    res = (PyArrayObject*)PyArray_SimpleNew(3,dims_res,NPY_INT32);

    // sort edges (in increasing order)
    float* data_attr = (float*)PyArray_DATA(attractive);
    npy_intp* index_attr = (npy_intp*)malloc(sizeof(npy_intp)*dims[0]*dims[1]*dims[2]*3);
    for (npy_intp i = 0; i < dims[0]*dims[1]*dims[2]*3; i++) {
        index_attr[i] = i;
    }
    quickSort(data_attr, index_attr, 0, dims[0]*dims[1]*dims[2]*3-1);

    float* data_rep = (float*)PyArray_DATA(repulsive);
    npy_intp* index_rep = (npy_intp*)malloc(sizeof(npy_intp)*dims[0]*dims[1]*dims[2]*9);
    for (npy_intp i = 0; i < dims[0]*dims[1]*dims[2]*9; i++) {
        index_rep[i] = i;
    }
    quickSort(data_rep, index_rep, 0, dims[0]*dims[1]*dims[2]*9-1);

    

    // 
    npy_intp p_attr = dims[0] * dims[1] * dims[2] * 3 - 1;
    npy_intp p_rep = dims[0] * dims[1] * dims[2 ]* 9 - 1;
    npy_intp i, j, k, d, n1, n2;
    while(1){
        if (data_attr[p_attr] < 0.0001){
            p_attr = 0;
        }
        if (data_rep[p_rep] < 0.0001){
            p_rep = 0;
        }
        if (p_attr == 0 && p_rep == 0){
            break;
        }
        // process an attract edge 
        if( data_attr[p_attr] > data_rep[p_rep] ){
            i = index_attr[p_attr] / (dims[1]*dims[2]*3);
            j = index_attr[p_attr] % (dims[1]*dims[2]*3) / (dims[2]*3);
            k = index_attr[p_attr] % (dims[2]*3) / 3;
            d = index_attr[p_attr] % 3;
            n1 = i * dims[1]*dims[2] + j*dims[2] + k;
            switch(d % 3) {
                case 0: n2 = n1 + dims[1]*dims[2]; break;
                case 1: n2 = n1 + dims[2]; break;
                default: n2 = n1 + 1; break;
            }
            if (!G_attr.is_connected(n1, n2) && !G_rep.is_connected(n1, n2)){
                G_attr.connect(n1, n2);
                G_rep.connect(n1, n2);
            }
            p_attr--;
        }
        // process a repulsive edge
        else{
            i = index_rep[p_rep] / (dims[1]*dims[2]*9);
            j = index_rep[p_rep] % (dims[1]*dims[2]*9) / (dims[2]*9);
            k = index_rep[p_rep] % (dims[2]*9) / 9;
            d = index_rep[p_rep] % 9;
            n1 = i * dims[1]*dims[2] + j*dims[2] + k;
            switch(d % 9) {
                case 0: n2 = n1 + dims[1]*dims[2]; break;
                case 1: n2 = n1 + dims[2]; break;
                case 2: n2 = n1 + 1; break;
                case 3: n2 = n1 + dims[1]*dims[2] + dims[2]; break;
                case 4: n2 = n1 + dims[1]*dims[2] - dims[2]; break;
                case 5: n2 = n1 + dims[1]*dims[2] + 1; break;
                case 6: n2 = n1 + dims[1]*dims[2] - 1; break;
                case 7: n2 = n1 + dims[2] + 1; break;
                default: n2 = n1 + dims[2] - 1; break;
            }
            if(!G_attr.is_connected(n1, n2)){
                G_rep.connect(n1, n2);
            }
            p_rep--;
        }
    }
    
    npy_intp root, size;
    for(npy_intp ind=0; ind<dims[0]*dims[1]*dims[2]; ind++){
        i = ind / (dims[1]*dims[2]);
        j = ind % (dims[1]*dims[2]) / dims[2];
        k = ind % dims[2];
        root = G_attr.root(ind);
        size = G_attr.cluster_size(root);
        if(size > min_sz){
            *(npy_intp *)PyArray_GETPTR3(res, i, j, k) = root + 1;
        }
        else{
            *(npy_intp *)PyArray_GETPTR3(res, i, j, k) = 0;
        }

        
    }

    return PyArray_Return(res);
}

static struct PyMethodDef mwsMethods[] = {
    {"mws2d_c", mws2d_c, METH_VARARGS, "mutex watershed for 2d image"},
    {"mws3d_c", mws3d_c, METH_VARARGS, "mutex watershed for 3d image"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mwsModule = {
    PyModuleDef_HEAD_INIT,
    "mws",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,  or -1 if the module keeps state in global variables. */
    mwsMethods,
    NULL,NULL,NULL,NULL
};

PyMODINIT_FUNC PyInit_mws(void) {
    import_array();
    return PyModule_Create(&mwsModule);
}