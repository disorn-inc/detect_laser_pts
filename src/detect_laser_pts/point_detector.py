#author: Jan Michalczyk
#script performing the image processing on the images acquired by the camera observing 
#laser pointers projections.

import roslib
roslib.load_manifest('detect_laser_pts')

import sys
import rospy
import cv2
import cv2.cv as cv
import numpy as np

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def drawLines(in_img, k_points):
    """
       function to draw horizontal lines on the image 
       helps to see the discrepancy between blobs

    """

    out_img = in_img
    (rows, cols, channels) = in_img.shape

    try:
        cv2.line( out_img, (int(k_points[0].pt[0]), 0), (int(k_points[0].pt[0] ), int(k_points[0].pt[1])), (255, 0, 0), 1, 8, 0)
        cv2.line( out_img, (int(k_points[1].pt[0]), 0), (int(k_points[1].pt[0] ), int(k_points[1].pt[1])), (255, 0, 0), 1, 8, 0)
        cv2.line( out_img, (int(k_points[0].pt[0]), int(k_points[0].pt[1])), ((int(k_points[0].pt[0]), rows - 1) ), (255, 0, 0), 1, 8, 0)
        cv2.line( out_img, (int(k_points[1].pt[0]), int(k_points[1].pt[1])), ((int(k_points[1].pt[0]), rows - 1) ), (255, 0, 0), 1, 8, 0)
    except IndexError:
        return out_img
    
    return out_img

def showRoi(in_img):
    """
       function to draw rectangular ROI on the image 
       helps to choose ROI properly

    """
    
    out_img = in_img
    (rows, cols, channels) = in_img.shape
    cv2.rectangle(out_img, (10, 30),  (530, 265), (0, 255, 0), 1)
    cv2.rectangle(out_img, (10, 310), (530, 450), (0, 255, 0), 1)

    return out_img

def removeNoise(in_img):
    """
       function to remove noise from image 
       whitens the ROIs with noise

    """

    #allocate a blank image (numpy array)
    out_img = 255*np.ones(in_img.shape, dtype=np.uint8)

    #switch indexes - rows are 'y' and cols are 'x' on images
    out_img[30:265, 10:530]  = in_img[30:265, 10:530]
    out_img[310:450, 10:530] = in_img[310:450, 10:530]

    return out_img

class PointsDetector:

    def __init__(self):
        self.image_pub = rospy.Publisher("image_points", Image, queue_size=10)
        cv2.namedWindow("PointsDetector", cv2.WINDOW_NORMAL)
        cv2.namedWindow("original_image", cv2.WINDOW_NORMAL)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_mono", Image, self.callback)

    def callback(self, data):

        #always enclose call to imgmsg_to_cv2() in try-catch to
        #catch conversion errors

        try:
            #cv_image is a numpy array
            cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
        except CvBridgeError, e:
            print e

        (rows, cols, channels) = cv_image.shape

        #threshold the image
        ret, cv_image_thresh = cv2.threshold(cv_image, 230, 255, cv2.THRESH_BINARY_INV)
        
        #remove noise from the original image
        cv_image_thresh = removeNoise(cv_image_thresh)
        
        #Set up parameters for blob detector
        params = cv2.SimpleBlobDetector_Params()

        #Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 100
        #Filter by Color
        params.filterByColor = True
        params.blobColor     = 0
        #Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity      = 0.6
        #Filter by Area
        params.filterByArea = True
        params.minArea      = 50
        #Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.1
        #Filter by Convexity - critical setting
        params.filterByConvexity = True
        params.minConvexity      = 0.5

        detector = cv2.SimpleBlobDetector(params)

        # Detect blobs
        keypoints = detector.detect(cv_image_thresh)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        cv_image_thresh_with_keypoints = cv2.drawKeypoints(cv_image_thresh, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv_image_thresh_with_keypoints = drawLines(cv_image_thresh_with_keypoints, keypoints)

        #cv_image_thresh_with_keypoints = drawLines(cv_image_thresh_with_keypoints, keypoints)
        cv2.imshow("PointsDetector", cv_image_thresh_with_keypoints)
        cv2.imshow("original_image", cv_image)
        cv2.waitKey(5)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image_thresh, "mono8"))
        except CvBridgeError, e:
            print e

def main( args ):
    
    rospy.init_node('PointsDetector', anonymous=True)
    pd = PointsDetector()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ...")

    cv2.destroyAllWindows()
