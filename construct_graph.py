from scipy.spatial.distance import pdist,squareform
import numpy as np

def intersection(lst1,lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def union(lst1,lst2):
    lst3 = list(set(lst1) | set(lst2))
    return lst3

def setdiff(lst1,lst2):
    lst3 = [value for value in lst1 if value not in lst2]
    return lst3

def dfs(v,marked,unmarked,A):
    tmp1 = A[v,:]

    n = len(tmp1)
    tmp2 = range(n)
    NNs = [i for i in tmp2 if tmp1[i] > 0]
    NNs = intersection(NNs,unmarked)
    marked = union(marked,NNs)
    unmarked = setdiff(unmarked,NNs)
    comp = NNs

    if len(NNs) > 0:
         N = len(NNs)
         for i in range(N):
             res = dfs(NNs[i],marked,unmarked,A)
             new_comp = res[0]
             marked = res[1]
             unmarked = res[2]
             comp = union(NNs,new_comp)
    return comp,marked,unmarked

def is_connected(A):
    n = len(A)
    marked = [0]
    unmarked = range(2,n)
    v_curr = 0
    res = dfs(v_curr,marked,unmarked,A)
    unmarked = res[2]
    connected = len(unmarked) == 0
    return connected
    
    
def DtoA(D,k,ind):
    N = len(D)
    A = np.zeros((N,N),dtype='int')
    for i in range(N):
        curr_ind = ind[range(1,(k+1)),i]
        for j in range(k):
            A[i,curr_ind[j]] = 1
            A[curr_ind[j],i] = 1
    return A

def is_conn(D,ind,k):
    A = DtoA(D,k,ind)
    connected = is_connected(A)
    return connected
    
def smallest_k(D,ind):
    k = 5
    connected = is_conn(D,ind,k)
    while not(connected):
        k = k + 1
        connected = is_conn(D,ind,k)
    return k
    
def find_comps(A):
    n = len(A)
    unmarked = range(n)
    count = 0
    comps = np.ones(n,dtype='int')
    while len(unmarked) > 0:
        marked = []
        v_curr = unmarked[0]
        marked = union([v_curr],marked)
        unmarked = setdiff(unmarked,[v_curr])
        res = dfs(v_curr,marked,unmarked,A)
        marked = res[1]
        unmarked = res[2]
        count = count + 1
        comps[marked] = count*np.ones(len(marked),dtype = 'int')

    return comps,count

def smallest_conn(D):
    """
    Recursively construct the SASNE graph using the
    disconnected-component procedure described in the paper.
    """

    N = D.shape[0]

    # ---------------------------------
    # Base cases
    # ---------------------------------
    if N == 0:
        return np.zeros((0, 0), dtype=int)

    if N == 1:
        return np.zeros((1, 1), dtype=int)

    ind = np.argsort(D, axis=0)

    k = smallest_k(D, ind)

    A = DtoA(D, k, ind)

    min_k = 5

    # ---------------------------------
    # Recursive refinement
    # ---------------------------------
    if k > min_k:

        k_disconnect = k - 1

        A_disconnect = DtoA(D, k_disconnect, ind)

        comps, Ncomps = find_comps(A_disconnect)

        # Components are labeled 1..Ncomps
        for comp_id in range(1, Ncomps + 1):

            comp_ind = [
                j
                for j in range(len(comps))
                if comps[j] == comp_id
            ]

            # Skip empty components
            if len(comp_ind) == 0:
                continue

            # Nothing to recurse on
            if len(comp_ind) == 1:
                continue

            sub_D = D[np.ix_(comp_ind, comp_ind)]

            A_sub = smallest_conn(sub_D)

            A[np.ix_(comp_ind, comp_ind)] = A_sub

    return A

def construct_graph(D):
    """
    D : NxN precomputed distance matrix
    """

    ind = np.argsort(D, axis=0)

    A = smallest_conn(D)
    #k = smallest_k(D,ind)
    #print(k)
    #A = DtoA(D,k,ind)
    N = len(D)
    W = np.zeros((N, N), dtype='float32')

    for i in range(N):
        for j in range(N):
            if A[i, j] > 0:
                W[i, j] = 1.0 / (1.0 + D[i, j]**2)

    return W
