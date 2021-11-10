
import cv2
import numpy as np
import math

def read_img(image_path):
    """ Reads image and converts to grayscale

    Args:
        image_path (r): Path to image.

    Returns:
        image: Grayscale image.
    """
    # reads image and converts RGB to gray
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    return image



# log transformation
def log_transformation(image_path, scaling_constant):
    """
    Log transformation function is s = c * log(1+r)
    where s is output and r is input pixel values

    Expands dark pixels and compresses bright pixels
    
    Args:
        scaling_constant: should be 255 / log(1+ abs(R)) where R
        is the maximum pixel value 
        image_path: path to image that will be transformed
     Returns:
        Image
    """



    image = read_img(image_path)
    new_img = np.zeros((image.shape[0], image.shape[1]))
    
    for i in image.shape[0]:

        for j in image.shape[1]:
            
            new_img[i,j] = scaling_constant * math.log(1 + image[i,j])
    
    return new_img  
    



def gamma_transformation(image_path, scaling_constant, gamma):
    """
    Gamma correction is used to manipulate brightness
    Transformation function is s = c * r ^ gamma
    fun fact: used for monitor correction
    
    Args:
        scaling_constant: should be 255 / log(1+ abs(R)) where R
        is the maximum pixel value 
        image_path: path to image that will be transformed
        gamma: constant used for correction, <1 results to brighter images
            and >1 results to darker images
    Returns:
        Image
    """
    
    image = read_img(image_path)
    corrected_img = np.zeros((image.shape[0], image.shape[1]))
    
    for i in image.shape[0]:

        for j in image.shape[1]:
            
            corrected_img[i,j] = scaling_constant * image[i,j]^gamma
    
    return corrected_img  
    

def image_binarize(image_path, threshold):
    """Image binarization (piecewise linear transformation)

    Args:
        image_path: path to image that will be transformed
        threshold: pixel value threshold as int (will be converted to list for 
                    non-monotonic transformation) 

    Returns:
        Image
    """

    image = read_img(image_path)
    binarized = np.zeros((image.shape[0], image.shape[1]))
    
    for i in image.shape[0]:

        for j in image.shape[1]:
            
            if image[i,j] < threshold:
            
                binarized_img[i,j] = 0
            
            else:

                binarized_img[i,j] = 255
    
    return binarized_img  



# negatives of image
def image_negative(image_path):

    image = read_img(image_path)
    neg = np.zeros((image.shape[0], image.shape[1]))
    
    for i in image.shape[0]:

        for j in image.shape[1]:
            
            neg[i,j] = 255-image[i,j]
    
    return neg
