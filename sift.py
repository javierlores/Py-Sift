#! /usr/bin/env python2.7

import csv
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.signal
import scipy.misc
import scipy.ndimage
import itertools
import math
import argparse

# Set how numpy handles errors
np.seterr(all='ignore')

NUM_OCTAVES = 4
NUM_SCALES = 5
OCTAVE_FACTOR = 2
SCALE_FACTOR = math.sqrt(2)
CONTRAST_THRESHOLD = 0.03
EDGE_THRESHOLD = 10
NUM_BINS = 8

def main():
    """ 
        This is the main function where the main logic of the program is held.

        Parameters
        ----------
        None
    """

    # Read in the arguments
    args = get_arguments()

    # Read in the image
    img = read_image(args.input)

    # Extract SIFT features from the img
    sift = Sift()
    features = sift.extract(img)

    # Write the sift features to a file
    if args.write:
        write_features(features, args.output)

    # Show the features overlayed on the original image
    if args.show:
        show_features(img, features)


def get_arguments():
    """ 
        This function retrieves the command line parameters passed into this script.

        Parameters
        ----------
        None
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='The file name of an image')
    parser.add_argument('output', type=str, help='The file name of the output')
    parser.add_argument('--write', action='store_true', help='The file name of the output')
    parser.add_argument('--show', action='store_true', help='The file name of the output')
    args = parser.parse_args()

    return args


def read_image(filename):
    """ 
        This function reads in a grayscale version of an image into a numpy array.

        Parameters
        ----------
        filename : str
            The file name of the image to be read.
    """

    return np.array(Image.open(filename).convert('L'))


def write_features(features, output_fn):
    """ 
        This function writes a feature pyramid to an output file.

        Parameters
        ----------
        output_fn : str
            The file name with which the features will be written.
        features : multi-dimensional array
            The features, organized as a pyramid, which will be written.
    """

    
    with open(output_fn+'.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)


        writer.writerow(['location', 'orientation', 'features'])
        for feature in features:
            writer.writerow(feature)


def show_features(img, features):
    """ 
        This function overlays a feature pyramid ontop of an image.

        Parameters
        ----------
        img : numpy array (2-D)
            The image to overlay the features ontop of.
        features : multi-dimensional array
            The features, organized as a pyramid, which will be displayed.
    """

    # Create our figure
    figure = plt.figure(figsize=(13, 9))
    plt.gray()

    # Plot img and features
    for point, orientation, feature_vector in features:
        axis = plt.gca()
        axis.add_patch(patches.Rectangle((point[1]-0.5, point[0]-0.5), 4, 4, facecolor='none', edgecolor='red'))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


class Sift():
    """ """
    def _init(self):
        """ """
        self.num_octaves = None
        self.num_scales = None
        self.octave_factor = None
        self.scale_factor = None
        self.contrast_threshold = None
        self.edge_threshold = None
        self.num_bins = 8


    def extract(self, img, 
                      num_octaves=NUM_OCTAVES, 
                      num_scales=NUM_SCALES, 
                      octave_factor=OCTAVE_FACTOR, 
                      scale_factor=SCALE_FACTOR, 
                      contrast_threshold=CONTRAST_THRESHOLD, 
                      edge_threshold=EDGE_THRESHOLD, 
                      num_bins=NUM_BINS):
        """ 
            This function extracts SIFT features from the image passed in.

            These options determine which/how many sift features will be extracted. 

            Parameters
            ----------
            num_octaves : int, optional
                The number of octaves to be used in the pyramid (default 4)
            num_scales : int, optional
                The number of scales to be used in the pyramid (default 5)
            octave_factor : int, optional
                The downsampling rate between each octave of the pyramid (default 2)
            scale_factor : float, optional
                The gaussian rate between each scale in an octave of the pyramid (default sqrt(2))
            contrast_threshold : float, optional
                The lower bound for low contrast features that will be removed (default 0.03)
            edge_threshold : int, optional
                The bound for which edge features with poor localization will be removed (default 10)
            num_bins : int, optional
                The number of bins to use for the orientation. This should be either 4 or 8
        """
        self.num_octaves = num_octaves
        self.num_scales = num_scales
        self.octave_factor = octave_factor
        self.scale_factor = scale_factor
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.num_bins = num_bins

        pyramid = self._scale_space_extrema_detection(img)
        pyramid = self._keypoint_localization(pyramid)
        pyramid = self._orientation_assignment(pyramid)
        pyramid = self._keypoint_descriptor(pyramid)

        features = self._collapse_pyramid(pyramid)   
        return features


    def _scale_space_extrema_detection(self, img):
        """ 
            This function performs the scale space extrema detection of sift.

            This consists of two steps: 
                1. Building the gaussian pyramid
                2. Detecting candidate keypoints.

            Parameters
            ----------
            img : numpy array (2-D)
                The image to perform the scale space extrema detection.
        """

        pyramid = self._build_gaussian_pyramid(img)
        pyramid = self._get_candidate_keypoints(pyramid)

        return pyramid


    def _build_gaussian_pyramid(self, img):
        """ 
            This function creates a gaussian pyramid for an image.

            Parameters
            ----------
            img : numpy array (2-D)
                The image around which to build the gaussian pyramid.
        """

        def pairwise(iterable):
            """ This is used to iterate over elements (s0, s1), (s1, s2)..."""
            a, b = itertools.tee(iterable)
            next(b, None)
            return itertools.izip(a, b)


        # Upsample the image by 2
        # This upsampling dramatically increases the runtime but
        # signifcantly increases the number of features retrieved.
        img = scipy.misc.imresize(img.copy(), float(2))

        pyramid = []
        for i in range(self.num_octaves):
            # Because of the upsampling, the first octave is SIGMA = 0.5*SCALE_FACTOR
            # All other octaves are SIGMA = OCTAVE*SCALE_FACTOR
            sigma = 0.5 if i == 0 else self.octave_factor**(i-1)

            # Downsample the image on all octaves except the first
            scale = scipy.misc.imresize(img, 1.0/self.octave_factor*i) if i != 0 else img

            # Normalize the image
            scale = scale/np.float32(255)

            # Create the octave
            octave = [scipy.ndimage.filters.gaussian_filter(scale, sigma*(self.scale_factor**(j+1))) for j in range(self.num_scales)]

            pyramid.append(octave)

        # Take the difference of gaussian(DoG) of the images
        pyramid = [[pair[1]-pair[0] for pair in pairwise(octave)] for octave in pyramid]

        return pyramid


    def _get_candidate_keypoints(self, pyramid):
        """ 
            This function gets the extremum keypoints from the DoG pyramid.

            The extremum keypoints are selected by looking at every pixel from every DoG image
            (that has a scale above and below it) and determining if that pixels intensity is 
            the maximum or minimum of its' 26 neighbors. The 26 neighbors consists of all pixels 
            within 3x3 boxes at the current scale (excluding the current pixel), the scale above, 
            and the scale below.

            Parameters
            ----------
            pyramid : multi-dimensional array
                The gaussian pyramid which contains the DoG images.
        """


        def compare(img, img_above, img_below, mode):
            """ 
                This function compares each pixel (x, y) to its neighbors in a 3x3 box in the scale above, 
                below, and around it and determines if the pixel (x, y) is a maximum or minimum in its
                neighborhood.

                Parameters
                ----------
                img : numpy array (2-D)
                    The current scale img
                img_above : numpy array (2-D)
                    The scale img above
                img_below : numpy array (2-D)
                    The scale img below
                mode : string, (min|max)
                    This value determines whether a minimum or maximum comparsion will be excuted

            """

            # Run a minimum comparison
            if mode == 'min':
                img_compare = scipy.ndimage.filters.minimum_filter(img, size=(3,3))
                img_above_compare = scipy.ndimage.filters.minimum_filter(img_above, size=(3,3))
                img_below_compare = scipy.ndimage.filters.minimum_filter(img_below, size=(3,3))

                comparison = (img == img_compare) & (img <= img_above_compare) & (img <= img_below_compare)

            # Run a maximum comparison
            if mode == 'max':
                img_compare = scipy.ndimage.filters.maximum_filter(img, size=(3,3))
                img_above_compare = scipy.ndimage.filters.maximum_filter(img_above, size=(3,3))
                img_below_compare = scipy.ndimage.filters.maximum_filter(img_below, size=(3,3))

                comparison = (img == img_compare) & (img >= img_above_compare) & (img >= img_below_compare)

            # Return the coordinates of each (x,y) that are computed from the mode
            coordinates = np.transpose(np.where(comparison))

            return coordinates


        # Get the candidate keypoints
        new_pyramid = []
        for octave_index, octave in enumerate(pyramid):
            new_octave = [[0, []] for i in range(self.num_scales-1)]
            # Only iterate over scales that have a scale above and below.
            for scale_index, scale in enumerate(octave[1:-1], start=1):
                img = scale
                img_above = octave[scale_index+1]
                img_below = octave[scale_index-1]

                # Check if the keypoint is a minimum or maximum             
                min_keypoints = compare(img, img_above, img_below, mode='min')
                max_keypoints = compare(img, img_above, img_below, mode='max')

                # Add the new image to the octave
                new_octave[scale_index][0] = img
                new_octave[scale_index][1].extend(min_keypoints.tolist() + max_keypoints.tolist())
            
            # Add the first and last octave
            new_octave[0][0] = octave[0]
            new_octave[-1][0] = octave[-1]
        
            # Add the new octave to the pyramid 
            new_pyramid.append(new_octave)

        return new_pyramid


    def _keypoint_localization(self, pyramid):
        """ 
            This function removes takes in the current candidate keypoints and 
            removes those that are of low contrast or are poorly localized along an edge.

            Parameters
            ----------
            pyramid : multi-dimensional array
                The gaussian pyramid which contains the DoG images and candidate keypoints.
        """

        pyramid = self._low_contrast_removal(pyramid)
        pyramid = self._poor_edge_localization_removal(pyramid)
        return pyramid
    

    def _low_contrast_removal(self, pyramid):
        """ 
            This function removes takes in the current candidate keypoints and 
            removes those that are of low contrast. A pixel is determined to be of low 
            contrast if 

            Parameters
            ----------
            pyramid : multi-dimensional array
                The gaussian pyramid which contains the DoG images and the candidate keypoints.
        """

        def compute_hessian(img, img_above, img_below):
            """ 
                This functions computes the second derivative hessian matrix of 
                the point located a x, y. In particular, the matrix is computed 
                using the finite difference formula where w[s][x][y] is the pixel in 
                scale 's' located at (x, y):

                    | h11 h12 h13 |
                H = | h12 h22 h23 |
                    | h13 h23 h33 |

                h11 = w[s+1][x][y] + w[s-1][x][y] - 2*w[s][x][y]
                h22 = w[s][x][y+1] + w[s][x][y-1] - 2*w[s][x][y]
                h33 = w[s][x+1][y] + w[s][x-1][y] - 2*w[s][x][y]

                h12 = (w[s+1][x][y+1] - w[s+1][x][y-1] - w[s-1][x][y+1] + w[s-1][x][y-1])/4
                h13 = (w[s+1][x+1][y] - w[s+1][x-1][y] - w[s-1][x+1][y] + w[s-1][x-1][y])/4
                h23 = (w[s][x+1][y+1] - w[s][x-1][y+1] - w[s][x+1][y-1] + w[s][x-1][y-1])/4

                Parameters
                ----------
                img : numpy array (2-D)
                    The current scale img
                img_above : numpy array (2-D)
                    The scale img above
                img_below : numpy array (2-D)
                    The scale img below
            """
            # Created shifted images
            x_diff_above, y_diff_above = self._compute_operation(img_above, axis=1, mode='difference')        
            x_diff_below, y_diff_below = self._compute_operation(img_below, axis=1, mode='difference')
 
            x_sum, y_sum = self._compute_operation(img, axis=1, mode='sum')
            x_diag, y_diag = self._compute_operation(img, axis=2, mode='sum')

            # calculate the hessian matrix
            hessian = np.zeros(img.shape + (3,3))

            h11 = img_above + img_below - 2*img
            h22 = y_sum - 2*img
            h33 = x_sum - 2*img

            h12 = (y_diff_above - y_diff_below)/4
            h13 = (x_diff_above - x_diff_below)/4
            h23 = (x_diag - y_diag)/4

            # Create 3x3 matrix
            hessian[..., 0, 0] = h11
            hessian[..., 0, 1] = h12
            hessian[..., 0, 2] = h13
            hessian[..., 1, 0] = h12
            hessian[..., 1, 1] = h22
            hessian[..., 1, 2] = h23
            hessian[..., 2, 0] = h13
            hessian[..., 2, 1] = h23
            hessian[..., 2, 2] = h33

            return hessian


        def compute_gradient(img, img_above, img_below):
            """ 
                This function computes the 3D gradient using the finite difference 
                formula where w[s][x][y] is the pixel in scale 's' located at (x, y):
                    | (w[s+1][x][y] - w[s-1][x][y]) / 2|
                    | (w[s][x][y+1] - w[s][x][y-1]) / 2|
                    | (w[s][x+1][y] - w[s][x-1][y]) / 2|

                Parameters
                ----------
                img : numpy array (2-D)
                    The current scale img
                img_above : numpy array (2-D)
                    The scale img above
                img_below : numpy array (2-D)
                    The scale img below
            """
            # Create shifted images
            x_diff, y_diff = self._compute_operation(img, axis=1, mode='difference')

            # Create gradient
            gradient = np.zeros(img.shape + (3,1))

            # Compute gradient
            gradient[..., 0, 0] = (img_above-img_below)/2
            gradient[..., 1, 0] = x_diff/2
            gradient[..., 2, 0] = y_diff/2

            return gradient


        def compute_contrast(img, x, y, gradient, u_prime):
            """ 
                Computes the contrast of a pixel (x, y) in an image.

                Parameters
                ----------
                img : numpy array
                    The current img
                x : int
                    The x location of the pixel
                y : int
                    The y location of the pixel
                gradient : 3x1 array
                    represents the 3D gradient at the pixel location
                u_prime : 
                    The offset of the pixel
            """

            return img[x][y] + 0.5*np.dot((gradient.T[0][::-1]), (u_prime.T[0][::-1]))


        # Iterate over all keypoints, eliminating those with low contrast
        new_pyramid = []
        for octave_index, octave in enumerate(pyramid):
            new_octave = [[0, []] for i in range(self.num_scales-1)]
            for scale_index, scale in enumerate(octave[1:-1], start=1):
                img = scale[0]
                img_above = octave[scale_index+1][0]
                img_below = octave[scale_index-1][0]
                keypoints = scale[1]

                # Compute the u_prime for the scale
                img_hessian = compute_hessian(img, img_above, img_below)
                img_gradient = compute_gradient(img, img_above, img_below)
                img_uprime = img_hessian/img_gradient

                new_keypoints = []
                for keypoint in keypoints:
                    # This is where we will store the next location of the keypoint
                    new_x = keypoint[0]
                    new_y = keypoint[1]
                    new_scale = scale_index
 
                    # If location of extremum is > 0.5 in any dimension,
                    # Then this extremum lies closed to a different point and 
                    # We hould perform interpolation around that point instead.

                    # Check x-dimension
                    if (abs(img_uprime[new_x][new_y][2][0]) > 0.5):
                        # If our new sample point is out of bounds, skip the keypoint
                        if (not self._is_valid(keypoint[0]+round(img_uprime[new_x][new_y][2][0]), keypoint[1], img)):
                            continue

                        # Compute new x location and new offset u'
                        new_x = int(keypoint[0] + round(img_uprime[new_x][new_y][2][0]))
                        img_uprime[new_x][new_y][2][0] = img_uprime[new_x][new_y][2][0] - round(img_uprime[new_x][new_y][2][0])

                    # Check y-dimension
                    if (abs(img_uprime[new_x][new_y][1][0]) > 0.5):
                        # If our new sample point is out of bounds, skip the keypoint
                        if (not self._is_valid(keypoint[0], keypoint[1]+round(img_uprime[new_x][new_y][1][0]), img)):
                            continue

                        # Compute new y location and new offset u'
                        new_y = int(keypoint[1] + round(img_uprime[new_x][new_y][1][0]))
                        img_uprime[new_x][new_y][1][0] = img_uprime[new_x][new_y][1][0] - round(img_uprime[new_x][new_y][1][0])

                    # Check sigma dimension
                    if (abs(img_uprime[new_x][new_y][0][0]) > 0.5):
                        # Sigma is in the scale above, compute new sigma, scale, and update u_prime
                        if (img_uprime[new_x][new_y][0][0] > 0):
                            new_scale = scale_index+1
                            sigma_above = 0.5*self.scale_factor**(scale_index+2) if octave_index == 0 else (self.octave_factor**(octave_index-1))*self.scale_factor**(scale_index+2)
                            sigma = 0.5*self.scale_factor**(scale_index+1) if octave_index == 0 else (self.octave_factor**(octave_index-1))*self.scale_factor**(scale_index+1)
                            
                            img_uprime[new_x][new_y][0][0] = sigma_above - sigma - img_uprime[new_x][new_y][0][0]
                        # Sigma is in the scale below, compute new sigma, scale, and update u_prime
                        else:
                            new_scale = scale_index-1
                            sigma_below = 0.5*self.scale_factor**(scale_index) if octave_index == 0 else (self.octave_factor**(octave_index-1))*self.scale_factor**(scale_index)
                            sigma = 0.5*self.scale_factor**(scale_index+1) if octave_index == 0 else (self.octave_factor**(octave_index-1))*self.scale_factor**(scale_index+1)
                            img_uprime[new_x][new_y][0][0] = sigma - sigma_below + img_uprime[new_x][new_y][0][0]

                    # Check the contrast threshold
                    contrast = compute_contrast(img, new_x, new_y, img_gradient[new_x][new_y], img_uprime[new_x][new_y])

                    # All extremum with a value < CONTRAST_THRESHOLD are unstable and are discarded
                    # All other extremum are added as keypoints
                    if (abs(contrast) >= self.contrast_threshold):
                        # Append to appropriate scale
                        if new_scale == octave_index and new_scale == scale_index:
                            new_keypoints.append((new_x, new_y))
                        else:
                            new_octave[new_scale][1].append((new_x, new_y))

                # Add the new image to the octave
                new_octave[scale_index][0] = img
                new_octave[scale_index][1].extend(new_keypoints)

            # Add the first and last octave
            new_octave[0] = octave[0]
            new_octave[-1] = octave[-1]
        
            # Add the new octave to the pyramid 
            new_pyramid.append(new_octave)

        return new_pyramid


    def _poor_edge_localization_removal(self, pyramid):
        """ 
            This function removes keypoints that are poorly localized along an edge.

            Parameters
            ----------
            pyramid : multi-dimensional array
                The gaussian pyramid which contains the DoG images and the candidate keypoints.
        """

        def compute_ratio(img):
            """ 
                This function computes the hessian ratio for each pixel in the scale image.

                Parameters
                ----------
                img : numpy array (2-D)
                    the image for which to compute the hessian ratio.
            """

            # Calculate second derivatives
            x, y = np.gradient(img)
            xx, xy = np.gradient(x)
            xy, yy = np.gradient(y)

            # Calculate determinant and trace
            det = xx*yy - xy**2
            tr = xx+yy

            # Calculate ratio
            return (tr**2) / det


        # Remove keypoints with poor edge localization
        new_pyramid = []
        for octave_index, octave in enumerate(pyramid):
            new_octave = [[0, []] for i in range(self.num_scales-1)]
            for scale_index, scale in enumerate(octave):
                img = scale[0]
                keypoints = scale[1]

                # Compute the hessian ratio and the threshold for that hessian ratio
                ratio = compute_ratio(img)
                threshold = ((self.edge_threshold+1)**2) / self.edge_threshold

                new_keypoints = []
                for keypoint in keypoints:
                    # Append the keypoint of it is less than than the threshold
                    if ratio is not None and ratio[keypoint[0]][keypoint[1]] < threshold:
                        new_keypoints.append(keypoint)

                # Add the new image to the octave
                new_octave[scale_index][0] = img
                new_octave[scale_index][1].extend(new_keypoints)

            # Add the new octave to the pyramid
            new_pyramid.append(new_octave)

        return new_pyramid


    def _orientation_assignment(self, pyramid):
        """ 
            This function performs the orientation assignment phase of SIFT.

            Parameters
            ----------
            pyramid : multi-dimensional array
                The gaussian pyramid which contains the DoG images and the candidate keypoints.
        """

        def loc_hist_peaks(hist):
            """ 
                This function performs the orientation assignment phase of SIFT.

                Parameters
                ----------
                hist : 1xnum_bins array
                    a histogram representing the orientations of the images
            """
            peak_val = max(hist)
            peaks = [self._get_orientation(index) for index, bin in enumerate(hist) if bin >= peak_val*.80]
            return peaks


        # Assign an orientation to every keypoint
        new_pyramid = []
        for octave_index, octave in enumerate(pyramid):
            new_octave = [[0, []] for i in range(self.num_scales-1)]
            for scale_index, scale in enumerate(octave):
                img = scale[0]
                keypoints = scale[1]

                # Compute the magnitude of every pixel and the orientation bin that pixel belongs to
                magnitudes = self._compute_magnitudes(img)
                orientation_bins = self._compute_bins(img)

                # The window size should be equal to the (size of the kernel)*1.5*(sigma*scale)
                # scipy's implementation of gaussian_filter uses a kernel size of truncate*standard_deviation+0.5
                # where truncate is when to truncate the standard deviation. default is == 4.0
                # So our window size is (4.0*sigma*scale)+0.5)*1.5(sigma*scale)
                sigma = 0.5*self.scale_factor**(scale_index+1) if octave_index == 0 else (self.octave_factor**(octave_index-1))*self.scale_factor**(scale_index+1)
                win_radius = math.ceil(4.0*sigma+0.5)

                new_keypoints = []
                for keypoint in keypoints:
                    # Define the bounds of the window to compute the orientation histogram
                    win_beg_x = keypoint[0]-win_radius if self._is_valid(keypoint[0]-win_radius, 0, img) else 0
                    win_beg_y = keypoint[1]-win_radius if self._is_valid(0, keypoint[1]-win_radius, img) else 0
                    win_end_x = keypoint[0]+win_radius if self._is_valid(keypoint[0]+win_radius, 0, img) else len(img)
                    win_end_y = keypoint[1]+win_radius if self._is_valid(0, keypoint[1]+win_radius, img) else len(img[0])

                    # Extract the magnitudes and orientations from the image
                    win_magnitudes = magnitudes[win_beg_x:win_end_x, win_beg_y:win_end_y]
                    win_orientation_bins = orientation_bins[win_beg_x:win_end_x, win_beg_y:win_end_y]
 
                    # Compute the histogram for the window
                    hist = self._compute_orientation_hist(win_magnitudes, win_orientation_bins)

                    # Locate peaks of the histogram
                    peaks = loc_hist_peaks(hist)

                    # For each peak, create a new keypoint
                    for orientation in peaks:
                        new_keypoints.append((keypoint, orientation))

                # Add the new image to the octave
                new_octave[scale_index][0] = img
                new_octave[scale_index][1].extend(new_keypoints)

            # Add the new octave to the pyramid
            new_pyramid.append(new_octave)

        return new_pyramid
                

    def _compute_orientation_hist(self, magnitude, bin):
        """ 
            This function creates the orientation histogram from the magnitude 
            image map and orientation bin map passed in. This function does so in 
            a vectorized manner to improve speed

            Parameters
            ----------
            pyramid : multi-dimensional array
                The gaussian pyramid which contains the DoG images and the candidate keypoints.
        """

        hist = []    
        for i in range(self.num_bins):
            bin_map = bin == i
            magnitude_map = bin_map * magnitude
            sum_map = np.sum(magnitude_map)

            hist.append(sum_map)

        return hist


    def _keypoint_descriptor(self, pyramid):
        """ 
            This function creates a feature vector for every keypoint.

            Parameters
            ----------
            pyramid : multi-dimensional array
                The gaussian pyramid which contains the DoG images and the candidate keypoints.
        """

        # Assign a feature vector to every keypoint
        new_pyramid = []
        for octave_index, octave in enumerate(pyramid):
            new_octave = [[0, []] for i in range(self.num_scales-1)]
            for scale_index, scale in enumerate(octave):
                img = scale[0]
                keypoints = scale[1]

                # Compute the magnitude of every pixel and the orientation bin that pixel belongs to
                magnitudes = self._compute_magnitudes(img)
                orientation_bins = self._compute_bins(img)

                new_keypoints = []
                for loc, orientation in keypoints:
                    # This is where we will store the feature vector
                    feature_vector = []

                    # The big window size is 16x16 so the radius should be 8
                    big_win_radius = 8

                    # To Discard keypoints too close to the image border
                    # We need to the set the bounds of how big our window is
                    x_lower_bound = big_win_radius
                    y_lower_bound = big_win_radius
                    x_upper_bound = len(img)-big_win_radius
                    y_upper_bound = len(img[0])-big_win_radius

                    # Now we check those bounds to ensure the x-value of the keypoint is valid
                    if (loc[0] <= x_lower_bound or loc[0] >= x_upper_bound):
                        continue
                    # Now we check those bounds to ensure the y-value of the keypoint is valid
                    if (loc[1] <= y_lower_bound or loc[1] >= y_upper_bound):
                        continue

                    # Define the bounds of the big window to compute the orientation histogram
                    big_win_beg_x = loc[0]-big_win_radius
                    big_win_beg_y = loc[1]-big_win_radius 
                    big_win_end_x = loc[0]+big_win_radius
                    big_win_end_y = loc[1]+big_win_radius

                    # Extract the window from the image magnitude map and orientation bins map
                    big_win_magnitudes = magnitudes[big_win_beg_x:big_win_end_x, big_win_beg_y:big_win_end_y]
                    big_win_orientation_bins = orientation_bins[big_win_beg_x:big_win_end_x, big_win_beg_y:big_win_end_y]

                    # Compute orientation histograms for each of the smaller 4x4 windows
                    for row in range(0, len(big_win_magnitudes)-3, 4):
                        for col in range(0, len(big_win_magnitudes[0])-3, 4):
                            small_win_radius = 2

                            # Define the bounds of the window to compute the orientation histogram
                            small_win_beg_x = row
                            small_win_beg_y = col
                            small_win_end_x = row+3
                            small_win_end_y = col+3

                            # Extract the smaller window from the large windows magnitude map and orientation bins map
                            small_win_magnitudes = big_win_magnitudes[small_win_beg_x:small_win_end_x, small_win_beg_y:small_win_end_y]
                            small_win_orientation_bins = big_win_orientation_bins[small_win_beg_x:small_win_end_x, small_win_beg_y:small_win_end_y]

                            # Compute the orientation histogram for the small window
                            hist = self._compute_orientation_hist(small_win_magnitudes, small_win_orientation_bins)

                            # Convert hist to feature vector
                            feature_vector.append(hist)

                    # Remove rotation dependence and illumination dependence
                    new_feature_vector = []
                    for hist in feature_vector:
                        new_hist = [0 for i in range(self.num_bins)]

                        for index, bin_val in enumerate(hist):
                            # Normalize feature vector
                            new_bin_val = float(bin_val) / ((big_win_radius*2)**2)

                            # Remove rotation dependence by subtracting the orientation of 
                            # the keypoint from the orientation of every pixel in the histogram
                            curr_orientation = self._get_orientation(index)
                            new_orientation = curr_orientation - orientation 
                            new_bin = self._get_bin(new_orientation)
                    
                            # Remove illumination dependence by thresholding
                            if new_bin_val > 0.2:
                                new_bin_val = 0.2

                            # renormalize feature vector
                            new_bin_val = float(new_bin_val) / ((big_win_radius*2)**2)

                            # Assign the bin to the new histogram
                            new_hist[new_bin] = new_bin_val

                        # Assign the histogram to the new feature vector
                        new_feature_vector.append(new_hist)

                    # Add the new keypoint
                    new_keypoints.append((loc, orientation, feature_vector))

                # Add the new image to the octave
                new_octave[scale_index][0] = img
                new_octave[scale_index][1].extend(new_keypoints)
            
            # Add the new octave to the pyramid
            new_pyramid.append(new_octave)

        return new_pyramid


    def _collapse_pyramid(self, pyramid):
        """ 
            This function collapses all the keypoints in a pyramid onto the original image.

            Parameters
            ----------
            pyramid : multi-dimensional array
                The gaussian pyramid which contains the DoG images and the candidate keypoints.
        """

        new_features = []
        for octave_index, octave in enumerate(pyramid):
            for scale_index, scale in enumerate(octave):
                # Calculate the factor to project the features onto the original image
                # The first octave is upsampled so calculate from the second octave
                factor = 1.0/2 if octave_index == 0 else self.octave_factor**(octave_index-1)
                for loc, orientation, feature_vector in scale[1]:
                    new_x = factor*loc[0]
                    new_y = factor*loc[1]
                    new_orientation = orientation
                    new_feature_vector = feature_vector

                    new_features.append(((new_x, new_y), new_orientation, new_feature_vector))
                
        return new_features


    def _is_valid(self, x, y, matrix):
        """ 
            This function is a conveinece function to check the bounds of a matrix.

            Parameters
            ----------
            x : int
                row on the image
            y : int
                column on the image
            matrix : numpy array (2-D)
                the img/matrix to check the bounds on

        """

        return (x < len(matrix) and x >= 0) and (y < len(matrix[0]) and y >= 0)


    def _get_bin(self, orientation):
        """ 
            This is a convenience function that returns a bin from an orientation.

            Parameters
            ----------
            orientation : int
                the orientation in degrees
        """

        if orientation % 360 == 0:
            return 0
        if orientation % 315 == 0:
            return 7
        if orientation % 270 == 0:
            return 6
        if orientation % 225 == 0:
            return 5
        if orientation % 180 == 0:
            return 4
        if orientation % 135 == 0:
            return 3
        if orientation % 90 == 0:
            return 2
        if orientation % 45 == 0:
            return 1


    def _get_orientation(self, bin):
        """ 
            This is a convenience function that returns an orientation from a bin number

            Parameters
            ----------
            bin : int
                a number in the range [0...num_bins-1]
        """

        orientations = [0, 45, 90, 135, 180, 225, 270, 315]
        return orientations[bin]


    def _compute_magnitudes(self, img):
        """ 
            Compute the magnitude of each pixel in the scale image.

            Parameters
            ----------
            img: numpy array (2-D)
               the image to compute the mangnitudes of
        """

        x_diff, y_diff = self._compute_operation(img, axis=1, mode='difference')
        return np.sqrt(x_diff**2 + y_diff**2)


    def _compute_bins(self, img):
        """ 
            Compute the orientation bin (for the histogram that will be created in
            orientation assignment) for each pixel.

            Parameters
            ----------
            img: numpy array (2-D)
               the image for which to compute the orientation bins
        """

        # Calculate the angle in radians
        x_diff, y_diff = self._compute_operation(img, axis=1, mode='difference')
        orientations = np.arctan2(y_diff, x_diff)

        # Round the orientations to the nearest 45
        orientations = orientations*180 / np.pi                # Convert to degrees
        orientations = orientations.astype('float32')          # Convert to float for rounding
        orientations = 45*np.round(orientations/45)            # Round to nearest 45 degree
        orientations = orientations.astype('uint8')            # Convert back to integer

        # Calculate the bin the orientation belongs in
        bin_maps = []
        for degrees in range(45, 360, 45): 
            bin_map = orientations == degrees
            bin_map = bin_map * orientations
            bin_map = bin_map / degrees
            bin_maps.append(bin_map)

        bins = np.zeros(img.shape)
        for index, bin in enumerate(bin_maps):
            bins = bins + bin*(index+1)

        return bins


    def _compute_operation(self, img, axis, mode):
        """ 
            This is a convenience function to calculate the the sum or difference between 
            img[x+1][y]-img[x-1][y] and img[x][y+1]-img[x][y-1]

            Parameters
            ----------
            img: numpy array (2-D)
                the image on which to perform the operation
            axis: (1|2)
                the axis along which to perform the operation
            mode: (sum|difference)
                the operation to perform on the image

        """
        # Compute along the x and y
        if axis == 1:
            if mode == 'sum':
                operation_x = np.r_[img[1:, :], [np.zeros(img.shape[1])]] + np.r_[[np.zeros(img.shape[1])], img[:-1, :]]
                operation_y = np.c_[img[:, 1:], np.zeros(img.shape[0])] + np.c_[np.zeros(img.shape[0]), img[:, :-1]]
            if mode == 'difference':
                operation_x = np.r_[img[1:, :], [np.zeros(img.shape[1])]] - np.r_[[np.zeros(img.shape[1])], img[:-1, :]]
                operation_y = np.c_[img[:, 1:], np.zeros(img.shape[0])] - np.c_[np.zeros(img.shape[0]), img[:, :-1]]

        # Compute along the diagonals
        if axis == 2:
            if mode == 'sum':
                operation_x = np.c_[np.zeros(img.shape[0]), np.r_[img[1:, :-1], [np.zeros(img.shape[1]-1)]]] \
                            + np.c_[np.r_[[np.zeros(img.shape[1]-1)], img[:-1, 1:]], np.zeros(img.shape[0])]
                operation_y = np.c_[np.r_[img[1:, 1:], [np.zeros(img.shape[1]-1)]], np.zeros(img.shape[0])]   \
                            + np.c_[np.zeros(img.shape[0]), np.r_[[np.zeros(img.shape[1]-1)], img[:-1, :-1]]]
            if mode == 'difference':
                operation_x = np.c_[np.zeros(img.shape[0]), np.r_[img[1:, :-1], [np.zeros(img.shape[1]-1)]]] \
                            - np.c_[np.r_[[np.zeros(img.shape[1]-1)], img[:-1, 1:]], np.zeros(img.shape[0])]
                operation_y = np.c_[np.r_[img[1:, 1:], [np.zeros(img.shape[1]-1)]], np.zeros(img.shape[0])]   \
                            - np.c_[np.zeros(img.shape[0]), np.r_[[np.zeros(img.shape[1]-1)], img[:-1, :-1]]]
 
        return operation_x, operation_y


if __name__ == "__main__":
    main()
