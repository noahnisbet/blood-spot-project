import numpy as np
import cv2
import imutils
import os
import logging
import glob

# set up to write to file

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped


def scan_contours(path_head_from = str, path_head_to = str):

    scale_frac = .5
    picture_list = []
    contour_points = []
    fail_list = []

    for filename in glob.glob(path_head_from+'*.jpeg'):
        picture_list.append(filename)

    for picture in picture_list:
	    base = os.path.basename(picture)
	    name =str(os.path.splitext(base)[0])

	    failed = False
	    img = cv2.pyrDown(cv2.imread(picture, cv2.IMREAD_UNCHANGED)) # read in

		# resize 
	    width = int(img.shape[1] * scale_frac)
	    height = int(img.shape[0] * scale_frac)
	    dim = (width, height)
	    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,0] # convert to hsv and select off h channel
	    img_gray = 255 * (img_gray / img_gray.max()) # rescale to range 0 to 255 to max out brightness

		# apply a binary threshold
	    _, threshed_img = cv2.threshold(img_gray, 35, 255, cv2.THRESH_BINARY) 
		
		# find contours and get the external one
	    contours, _ = cv2.findContours(threshed_img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		# find contour with max area
	    max_contour = max(contours, key=lambda x: cv2.contourArea(x))

		# draw max contour over image
		# cv2.drawContours(img, [max_contour], -1, (255, 255, 0), 1)

		# simplify contours
	    epsilon = 0.01*cv2.arcLength(max_contour,True)
	    approx = cv2.approxPolyDP(max_contour,epsilon,True)
		
	    if len(approx)==4:
		    contour_points.append(approx[0])
	    else:
	        failed = True

		#formatting coordinates for skew transform
	    points = []
	    for coordinates in approx: 
		    points.append(tuple(coordinates[0]))
	    points = np.array(points)

		# skew transform
	    warped = four_point_transform(img, points)
	    resized = cv2.resize(warped, (320,820), interpolation = cv2.INTER_AREA)
	    spots = resized[110:175,0:320]
	    spot1 = spots[5:62,7:72]
	    spot2 = spots[5:62,68:133]
	    spot3 = spots[5:62,130:195]
	    spot4 = spots[5:62,190:255]
	    spot5 = spots[5:62,252:317]

	    if not failed:
		    cv2.imwrite(os.path.join(path_head_to, name + ".jpeg"), warped)
		    cv2.imwrite(os.path.join(path_head_to, name + "(all_spots).jpeg"), spots)
		    cv2.imwrite(os.path.join(path_head_to, name + "(spot1).jpeg"), spot1)
		    cv2.imwrite(os.path.join(path_head_to, name + "(spot2).jpeg"), spot2)
		    cv2.imwrite(os.path.join(path_head_to, name + "(spot3).jpeg"), spot3)
		    cv2.imwrite(os.path.join(path_head_to, name + "(spot4).jpeg"), spot4)
		    cv2.imwrite(os.path.join(path_head_to, name + "(spot5).jpeg"), spot5)

	    else:
	        fail_list.append(name)
	        name = name + "(FAILED).jpeg"
	        cv2.imwrite(os.path.join(path_head_to, name), warped)
    
    return fail_list


if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--input_folder', '-in', type=str, required=True, help="The file path for the directory of input photos.")
	parser.add_argument('--output_folder', '-out', type=str, required=True, help="The file path for the directory of output photos.")
	parser.add_argument('--log-file', '-lf', type=str, default='./log.txt', help="The log file path")
	args = parser.parse_args()

	logging.basicConfig(filename='log.txt', level=logging.INFO)

	fail_list = scan_contours(args.input_folder,args.output_folder)
    
	logging.info(" FAILED TO TRANSFORM: " + ", ".join(fail_list))