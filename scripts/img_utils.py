
import cv2

def compute_image_resize(image, width = None, height = None):
    # From https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    
    return dim
 
    
def img_resize(img_path,size=None,ratio=None):
    if not size:
        size=[0,480]
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    dim = compute_image_resize(img, height=size[1])
    img = cv2.resize(img, dim, interpolation =  cv2.INTER_AREA)
    return img