#!/bin/bash
import numpy as np
import argparse
import cv2

# http://stackoverflow.com/questions/16665742/a-good-approach-for-detecting-lines-in-an-image
# http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

# process, threshol
# CV_8UC1 is a 8bit one channel color
CVINT = "uint8"


def main():
    parser = argparse.ArgumentParser(description='get config input')
    parser.add_argument('picture_path', metavar='path', type=str,
                        help='picture file path')
    args = parser.parse_args()
    img_file_path = args.picture_path
    image = cv2.imread(img_file_path, cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.GaussianBlur(image, (21, 21), 0)
    #cv2.equalizeHist(img_blur, img_blur)


    cv2.imshow("image", img_blur)
    cv2.waitKey(0)
    _, white_img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    #image = white_img
    _, white_img_blur = cv2.threshold(img_blur, 180, 255, cv2.THRESH_BINARY)
    image = white_img_blur
    skel = np.zeros(image.shape, dtype="uint8")
    temp = np.empty(image.shape, dtype="uint8")
    eroded = np.empty(image.shape, dtype="uint8")
    # structure elements are just numpy arrays, this one looks like a cross (a plus symbol)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    print(element)
    done = False
    show = False

    while not done:
        cv2.erode(image, element, eroded)
        if show:
            print("eroded")
            cv2.imshow("image", eroded)
            cv2.waitKey(0)
        cv2.dilate(eroded, element, temp)
        if show:
            print("dilated temp")
            cv2.imshow("image", temp)
            cv2.waitKey(0)
        cv2.subtract(image, temp, temp)
        if show:
            print("subtracted temp")
            cv2.imshow("image", temp)
            cv2.waitKey(0)
        skel = cv2.bitwise_or(skel, temp, skel)
        if show:
            print("subtracted skel")
            cv2.imshow("image", skel)
            cv2.waitKey(0)
        image = np.copy(eroded)

        nonzero = cv2.countNonZero(image)
        done = (nonzero == 0)
        print(nonzero)
        # cv2.imshow("images", skel)
        # cv2.waitKey(0)
    cv2.imshow("images", skel)
    cv2.waitKey(0)

    adaptive_white = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 21, 2)
    # bgr min bgr max
    boundaries = [([200, 200, 200], [255, 255, 255])]

    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype=CVINT)
        upper = np.array(upper, dtype=CVINT)

        # mask = cv2.inRange(image, lower, upper)
        # output = cv2.bitwise_and(image, image, mask=mask)

        # show the images
        # cv2.imshow("images", np.hstack([image, output]))
        cv2.imshow("images", white_img)
        cv2.waitKey(0)
        cv2.imshow("images", white_img_blur)
        cv2.waitKey(0)
    pass


if __name__ == "__main__":
    # run main program
    main()
