import numpy as np


# Maxpooling Function
def maxpooling_2d(arr: np.ndarray, size: int, stride: int) -> np.ndarray:
    """
    This function is used to perform maxpooling on the input array of 2D matrix(image). 
    Pooling is mainly used to reduce the size of the input matrix.
    
    Args:
        arr: numpy array
        size: size of pooling matrix
        stride: the number of pixels shifts over the input matrix
    
    Returns:
        numpy array of maxpooled matrix
    """
    arr = np.array(arr)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("The input array is not a square matrix")
    i = 0
    j = 0
    mat_i = 0
    mat_j = 0

    # compute the shape of the output matrix
    maxpool_shape = (arr.shape[0] - size) // stride + 1
    # initialize the output matrix with zeros of shape maxpool_shape
    updated_arr = np.zeros((maxpool_shape, maxpool_shape))

    while i < arr.shape[0]:
        if i + size > arr.shape[0]:
            # if the end of the matrix is reached, break
            break
        while j < arr.shape[1]:
            # if the end of the matrix is reached, break
            if j + size > arr.shape[1]:
                break
            # compute the maximum of the pooling matrix
            updated_arr[mat_i][mat_j] = np.max(arr[i : i + size, j : j + size])
            # shift the pooling matrix by stride of column pixels
            j += stride
            mat_j += 1

        # shift the pooling matrix by stride of row pixels
        i += stride
        mat_i += 1

        # reset the column index to 0
        j = 0
        mat_j = 0

    return updated_arr


# Averagepooling Function
def avgpooling_2d(arr: np.ndarray, size: int, stride: int) -> np.ndarray:
    """
    This function is used to perform avgpooling on the input array of 2D matrix(image)
    
    Args:
        arr: numpy array
        size: size of pooling matrix
        stride: the number of pixels shifts over the input matrix
    
    Returns:
        numpy array of avgpooled matrix
    """
    arr = np.array(arr)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("The input array is not a square matrix")
    i = 0
    j = 0
    mat_i = 0
    mat_j = 0

    # compute the shape of the output matrix
    avgpool_shape = (arr.shape[0] - size) // stride + 1
    # initialize the output matrix with zeros of shape avgpool_shape
    updated_arr = np.zeros((avgpool_shape, avgpool_shape))

    while i < arr.shape[0]:
        # if the end of the matrix is reached, break
        if i + size > arr.shape[0]:
            break
        while j < arr.shape[1]:
            # if the end of the matrix is reached, break
            if j + size > arr.shape[1]:
                break
            # compute the average of the pooling matrix
            updated_arr[mat_i][mat_j] = int(np.average(arr[i : i + size, j : j + size]))
            # shift the pooling matrix by stride of column pixels
            j += stride
            mat_j += 1

        # shift the pooling matrix by stride of row pixels
        i += stride
        mat_i += 1
        # reset the column index to 0
        j = 0
        mat_j = 0

    return updated_arr
