
import cv2
import numpy as np
from skimage import transform as trans


def preprocess(img1,landmark=None):
  M = None
  image_size = [112,112]
  warped=None
  if landmark is not None:
    assert len(image_size)==2
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    if image_size[1]==112:
      src[:,0] += 8.0
    dst = landmark.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    #print(img1.shape)
    warped = cv2.warpAffine(img1,M,(image_size[1],image_size[0]), borderValue = 0.0)
  return warped