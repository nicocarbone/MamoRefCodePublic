# Auxiliary functions
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import ndimage
from tqdm import tqdm

def loadMatStruct (fileName, keyName, index=1):
    '''
    Load MatLab generated MC simulations
    
    fileName: input filename
    keyName: dictionary key to load
    index: index to load inside the keyName
    
    '''
    rawMat = sio.loadmat(fileName, squeeze_me=True)[keyName]
    imgArrMat = []

    for i in range(np.shape(rawMat)[0]):
        for j in range(np.shape(rawMat)[1]):
            imgArrMat.append(rawMat[i,j].item()[index])

    return np.asarray(imgArrMat), np.shape(np.asarray(imgArrMat))

def cropAroundPosition(inImage, xCenter, yCenter, xSize, ySize):
    '''
    Crop an image around a (x,y) center
    
    inImage: input image
    xCenter, yCenter: center around cropping
    xSize, ySize: cropping size    
    '''    
    
    import numpy as np
    
    imageCropped = np.full((xSize, ySize), fill_value=np.nan)
    xSizeImage, ySizeImage = np.shape(inImage)
    
    for x in range(xSize):
        for y in range(ySize):
            xDiff = int(xSize/2-x)
            yDiff = int(ySize/2-y)            
            
            if xCenter-xDiff >= 0 and yCenter-yDiff >= 0: 
                if xCenter-xDiff < xSizeImage and yCenter-yDiff < ySizeImage:
                    imageCropped[x][y] = inImage[xCenter-xDiff][yCenter-yDiff]

    return imageCropped

def gaussian2D(xSize, ySize, x0In, y0In, sigma, muu, contrast=10, baseValue=0):
    '''
    Generate 2D gaussian profile around (x0In, y0In) and store in an array of size (xSize, ySize)
    '''
    #import numpy as np

    x0 = xSize/2-x0In
    y0 = ySize/2-y0In

    # Initializing value of x-axis and y-axis
    # in the range -1 to 1
    x, y = np.meshgrid(np.linspace(-ySize/2+y0, ySize/2+y0, ySize),
                       np.linspace(-xSize/2+x0, ySize/2+x0, xSize))
    dst = np.sqrt(x*x+y*y)

    # Calculating Gaussian array
    gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2)))
    if contrast != 0:
        gauss = (np.abs(gauss-1/contrast) + (gauss-1/contrast)) / 2
        gauss[gauss == 0] = baseValue
        gauss = gauss/np.max(gauss)
    return gauss

def fftGauss(in_arrayOriginal, kernel_size):
    '''
    Perform FFR Gauss low-pass filtering

    in_array: input image
    kernel_size: kernel size for smoothing
    '''
    if kernel_size > 0:
        in_NaNs = np.isnan(in_arrayOriginal)
        in_array = np.copy(in_arrayOriginal)
        in_array[np.isinf(in_array)] = 0
        in_array[np.isnan(in_array)] = 0

        # FFT filter
        im_fft = np.fft.fft2(in_array)
        im_rfft_filtered = ndimage.fourier_gaussian(im_fft, kernel_size)
        im_filtered = np.fft.ifft2(im_rfft_filtered)

        # Power spectrum
        pwr_spectrum = abs(np.fft.fftshift(im_fft))**2
        pwr_spectrum_filtered = abs(np.fft.fftshift(im_rfft_filtered))**2

        # Re-normalization
        sum_ratio = in_array.sum()/im_filtered.sum()
        im_filtered = im_filtered*sum_ratio
        del in_array

        im_output = np.full(np.shape(im_filtered), fill_value=np.NaN)
        im_output[~in_NaNs] = im_filtered[~in_NaNs]
        del im_filtered, in_NaNs

        return im_output, pwr_spectrum, pwr_spectrum_filtered
    else:
        return in_arrayOriginal, 0, 0


def fftGauss1D(signal, cutoff_freq):
    # Compute the 1D FFT of the signal
    fft_signal = np.fft.fft(signal)

    # Create a frequency domain representation
    freq = np.fft.fftfreq(len(signal))

    # Apply a low-pass filter by zeroing out the high-frequency components
    fft_signal[np.abs(freq) > cutoff_freq] = 0

    # Compute the inverse FFT to obtain the filtered signal
    filtered_signal = np.fft.ifft(fft_signal)

    # Return the real part of the filtered signal (ignoring imaginary part)
    return np.real(filtered_signal)


def movingAverage(arr, window_size):
    # Pad the array at the beginning and end with zeros
    padded_arr = np.pad(arr, (window_size // 2, window_size // 2), mode='edge')

    # Create a sliding window view of the padded array
    window_view = np.lib.stride_tricks.sliding_window_view(padded_arr, window_shape=(window_size,))

    # Compute the average along the window axis
    smoothed_arr = np.mean(window_view, axis=1)

    return smoothed_arr


def maxPosCm(in_array, smooth=40, threshold=0.6):
    '''
        Find maximum of the 2d array by finding the center of
        mass of a discretized image

        in_array: input array
        smooth: size of the kernel for the gaussian filter used in
        smoothing
        threshold [0-1]: factor of the maximum to consider for the
        discretization
    '''
    in_array = np.nan_to_num(in_array)

    if len(in_array) > 0:
        # Smooth image
        if smooth > 0:
            im_temp = fftGauss(in_array, smooth)[0]
        else:
            im_temp = in_array

        del in_array

        # Find the maximim of the smoothed image
        max_im = np.max(im_temp)

        # Discretize the image in two levels
        im_temp[im_temp < max_im*threshold] = 0
        im_temp[im_temp > max_im*threshold] = 1

        # Find the center of mass
        #cm = ndimage.measurements.center_of_mass(im_temp)
        cm = ndimage.center_of_mass(im_temp)
        return[int((cm[1])), int((cm[0]))]


def recon2D(images, centers, weights=[], method="mean", reconShape=[0,0]):
    '''
    Topografic 2D reconstrucion
    
    images: normalized and processed images
    centers: centers position of images
    weights: matrix of weights for the averaging of pixels. Should be of the same size as each image.
    
    Returns the final reconstructed image and 2D array with the number of images that constribute to each pixel.
    '''
    
    xSize,ySize = np.shape(images[0])
    
    maxCenters = int(np.amax(centers))  
    if reconShape[0] == 0 or reconShape[1] == 0:
        reconShape = (np.shape(images[0])[0]+maxCenters - int(xSize/2), np.shape(images[0])[1]+maxCenters- int(ySize/2)) #TODO: Can be improved...
    
    imageReconBackLevel = 0

    if method == "mean":
        imageReconTemp = np.full(reconShape, fill_value=imageReconBackLevel, dtype=float)
        imageReconCount = np.zeros(reconShape)
    
    elif method == "median":
        imageReconTemp = np.full((np.shape(images)[0],reconShape[0],reconShape[1]), fill_value=np.nan, dtype=float)
    
       
    
    if np.shape(weights)[0]<1:
        weights = np.full((xSize,ySize), fill_value=1, dtype=float)
        
    for i, image in tqdm(enumerate(images)):
        centerX = int(centers[i][1])
        centerY = int(centers[i][0])
        #xSize,ySize = np.shape(image)
       
        # For each pixel in imageDiv
        for x in np.arange(xSize):
            for y in np.arange(ySize):
                if ~np.isnan(image[x][y]):
                    posX = centerX + x - int(xSize/2)
                    posY = centerY + y - int(ySize/2)
                    
                    if method == "mean":
                        try:
                            imageReconTemp[posX][posY] += image[x][y]*weights[x][y]
                            imageReconCount[posX][posY] += weights[x][y]
                        except Exception:
                            print("Error in image {}... skipping.".format(i))
                            pass
                    elif method == "median":
                        try:
                            imageReconTemp[i][posX][posY] = image[x][y]
                        except Exception:
                            print("Error in image {}... skipping.".format(i))
                            pass
                        
    if method == "mean":                
        imageRecon = imageReconTemp/imageReconCount
        return imageRecon, imageReconCount
    if method == "median":
        imageRecon = np.nanmedian(imageReconTemp, axis=0)
        return imageRecon, []
    
def O2Sat (HbOCon, HbRCon):
    if HbOCon < 0: HbOCon = 0
    if HbRCon < 0: HbRCon = 0
    O2SatPerc = np.round(HbOCon/(HbOCon+HbRCon)*100)
    return O2SatPerc

def FWHM(X,Y):
    '''https://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak
    '''  
    half_max = max(Y) / 2.
    
    #find when function crosses line half_max (when sign of diff flips)
    #take the 'derivative' of signum(half_max - Y[])
    d = np.sign(half_max - np.array(Y[0:-1])) - np.sign(half_max - np.array(Y[1:]))
    
    #plot(X[0:len(d)],d) #if you are interested
    #find the left and right most indexes
    left_idx = np.where(d > 0)[0]
    right_idx = np.where(d < 0)[-1]
    
    return X[right_idx] - X[left_idx] #return the difference (full width)
