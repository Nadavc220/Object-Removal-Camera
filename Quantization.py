# =========== Imports =========== #
import warnings
import numpy as np
from sklearn.cluster import KMeans as km

# =========== Constants =========== #
GRAYSCALE = 1
RGB = 2
PIXEL_AXIS = 2
NUM_OF_RGB_VALS = 3
MAX_GRAY_VAL = 255
NUM_OF_GRAY_VALS = 256
Y_CHANNEL = 0
FIRST_ITERATION = 1
MAX_FLOAT_GRAY_VAL = 1
MIN_GRAY_VAL = 0


# =========== Globals =========== #

# A matrix which is used to yiq-Rgb conversions
trans_mat = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])

# =========== Main Functions =========== #


def quantize_rgb_image(im_orig, n_quant, n_iter):
    """
    Converts a given RGB image color spectrum to be n_quant rgb colors in the most optimal way.
    :param im_orig: the given image.
    :param n_quant: number of colors in output image.
    :param n_iter: how many optimizations should be processed.
    :return: [im, error] where im is a quantized image and error is a list of error values from all
    optimization iterations.
    """
    # in this algorithm we will use the k-means algorithm, to help us with that we
    #  shall change the images shape to 2 dimensions
    width, height, color_num = im_orig.shape
    resh_im = np.reshape(im_orig, (width * height, color_num))

    # calculating k-means
    partition, quants, error = quantize_rgb_list(resh_im, n_quant, n_iter)

    # reshaping image and updating color values
    quant_im = np.reshape(partition[quants], (width, height, color_num))
    return [quant_im, np.array(error)]


def quantize_rgb_list(rgb_list, n_quant, n_iter):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        error = []
        n_quant = min(n_quant, len(rgb_list))

        # getting initial partition and quants
        partition, quants = get_kmeans_initial_partition(rgb_list, n_quant, error)

        # optimizing partition and quants
        partition, quants = optimize_kmeans_quantization_vals(rgb_list, partition, n_quant, n_iter, error)

        return partition, quants, error


def get_kmeans_initial_partition(resh_im, n_quant, error):
    """
    makes initial partition and quants using k_means for rgb quantization algorithm.
    :param resh_im: the image to quantize.
    :param n_quant: number of colors in image.
    :param error: an array holding error values for partition and quants.
    :return: the initial partition and quants.
    """
    k_means = km(n_clusters=n_quant, max_iter=1, n_init=1)
    quants = k_means.fit_predict(resh_im)
    partition = k_means.cluster_centers_
    error.append(k_means.inertia_)
    return partition, quants


def optimize_kmeans_quantization_vals(resh_im, partition, n_quant, n_iter,  error):
    """
    Optimizes the partition and quants with at most n_iter iteration of rearrangement.
    :param resh_im: the image quantized.
    :param partition: the current partition.
    :param n_quant: number of colors in output.
    :param n_iter: number of iterations to optimize (at most).
    :param error: an array holding error values for quants and partition.
    :return: optimized partition and quants.
    """
    quants = []
    for i in range(n_iter - 1):
        k_means = km(n_clusters=n_quant, init=partition, max_iter=1, n_init=1)
        quants = k_means.fit_predict(resh_im)
        new_partition = k_means.cluster_centers_
        # checking for convergence
        if np.array_equal(new_partition, partition):
            break
        partition = new_partition
        error.append(k_means.inertia_)
    return partition, quants