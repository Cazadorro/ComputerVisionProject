import cv2
import numpy as np


def run():
    MUL = 1
    image = cv2.imread("skeleton.png", cv2.IMREAD_GRAYSCALE)
    src_pts = np.array([[24, 601], [457, 597], [404, 493], [79, 491]])
    dst_pts = np.array([[2* MUL, 58 * MUL], [23 * MUL, 58 * MUL] ,[23 * MUL, 47 *MUL], [2 * MUL, 47 *MUL]])

    src = src_pts
    dst = dst_pts

    translate = np.array([[1,0,0],
                          [0,1, 0 ],
                          [0,0,1]])

    h, status = cv2.findHomography(src, dst)
    im_dst = cv2.warpPerspective(image, h, (24 * MUL, 60 *MUL))
    #im_dst[:10, :] = 0

    cv2.imwrite("test_line.jpg", im_dst)


if __name__ == "__main__":
    # run main progra
    run()
