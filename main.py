#!/bin/bash
import numpy as np
import argparse
import cv2

# http://stackoverflow.com/questions/16665742/a-good-approach-for-detecting-lines-in-an-image
# http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

# process, threshol
# CV_8UC1 is a 8bit one channel color
CVINT = "uint8"

def parse_args():
    parser = argparse.ArgumentParser(description='get config input')
    parser.add_argument('image_path', metavar='path', type=str,
                        help='picture file path')
    args = parser.parse_args()
    return args

def get_skeleton(image):
    skel = np.zeros(image.shape, dtype="uint8")
    temp = np.empty(image.shape, dtype="uint8")
    eroded = np.empty(image.shape, dtype="uint8")
    # structure elements are just numpy arrays, this one looks like a cross (a plus symbol)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        #create an eroded version of the image
        cv2.erode(image, element, eroded)
        #dilate back out
        cv2.dilate(eroded, element, temp)
        # subtract the dilated version from the original version of the image
        cv2.subtract(image, temp, temp)
        # take the skelliton of the image and or it with the dilated subtraction
        cv2.bitwise_or(skel, temp, skel)
        # copy the eroded version to image.
        image = np.copy(eroded)
        # check if nonzero, if zero, we know we have only one pixel thick lines
        nonzero = cv2.countNonZero(image)
        done = (nonzero == 0)
        print(nonzero)
    return skel

def main():
    # getting initial image
    args = parse_args()
    image_path = args.image_path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (11, 11), 0)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    # img_blur = cv2.GaussianBlur(image, (3, 3), 0)
    _, white_image = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY)
    cv2.imshow("image", white_image)
    cv2.waitKey(0)
    white_image = cv2.GaussianBlur(white_image, (5, 5), 10)
    cv2.imshow("image", white_image)
    cv2.waitKey(0)
    _, white_image = cv2.threshold(white_image, 170, 255, cv2.THRESH_BINARY)
    cv2.imshow("image", white_image)
    cv2.waitKey(0)
    skeleton = get_skeleton(white_image)
    cv2.imshow("image", skeleton)
    cv2.waitKey(0)
    cv2.imwrite("skeleton", skeleton)



if __name__ == "__main__":
    # run main program
    main()
