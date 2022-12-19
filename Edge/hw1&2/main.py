import cv2
img = cv2.imread('le.jpg')

laplacian_edge = ???
canny_edge = ???
def vis(x):
    cv2.imshow('a',x)
    cv2.waitKey(3000)
vis(laplacian_edge)
vis(canny_edge)