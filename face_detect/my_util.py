
import cv2
import numpy as np
from skimage import transform as trans
import os
import glob
from multiprocessing.pool import ThreadPool
import pickle


def race_preprocess(img, landmark=None, **kwargs):
  if isinstance(img, str):
    img = read_image(img, **kwargs)
  M = None
  image_size = [400, 400]
  if landmark is not None:
    assert len(image_size)==2
    src = np.array([
      [157.5944, 180],
      [242.4064, 180],
      [199.76024, 221.28209],
      [162.78812, 274.4534],
      [237.37758, 274.14816] ], dtype=np.float32 )
    dst = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
    return warped


def get5landmark(preds):
  print(preds.shape)
  points=[]
  points.append((preds[36]+preds[39])/2)
  points.append((preds[42]+preds[45])/2)
  points.append(preds[30])
  points.append(preds[48])
  points.append(preds[54])
  points=np.array(points).astype(np.float32)
  return points



def createFolder(directory):
  try:
    if not os.path.exists(directory):
      os.makedirs(directory)
  except OSError:
    print ('Error: Creating directory. ' +  directory)


def drawing_landmark(image,landmark):
        #if landmark == None: return None
        radius = 3
        color  = (0,255,0)
        thickness = -1
        for x,y in landmark:
            cv2.circle(image,(x,y),radius,color, thickness)


class Saver():
  def __init__(self, foldername, split_length):
    self.split_length = split_length
    self.foldername = os.path.dirname(foldername)
    # print(self.foldername)
    if not os.path.exists(self.foldername):
      os.makedirs(self.foldername)

    self.feats = []
    self.pkl_counter = 0
    self.feat_len = 0

  def push(self, feat):
    self.feats.append(feat)
    self.feat_len += feat.shape[0]
    if self.feat_len >= self.split_length:
      filepath = os.path.join(self.foldername, '%04d.pkl'%self.pkl_counter)
      f = open(filepath, 'wb')
      self.feats = np.concatenate(self.feats, axis=0)
      pickle.dump(self.feats, f)
      f.close()
      print('Feature saved:', filepath, 'Shape:', self.feats.shape)
      # post process
      self.feats = []
      self.pkl_counter += 1
      self.feat_len =0

  def finish(self):
    if self.feat_len>0:
      filepath = os.path.join(self.foldername, '%04d.pkl'%self.pkl_counter)
      f = open(filepath, 'wb')
      self.feats = np.concatenate(self.feats, axis=0)
      pickle.dump(self.feats, f)
      f.close()
    self.pkl_counter += 1

def load_matrix(path, split=50000):
  print(path)
  splits = len(glob.glob('./%s_data/*.pkl'%path))
  result = []
  for i in range(splits):
    f = open('./%s_data/%04d.pkl'%(path,i),'rb')
    mtx = pickle.load(f)
    result.append(mtx)
    f.close()
  result = np.concatenate(result, axis=0)
  return result 


def get_image(image_name):
  img = cv2.imread(image_name)
  return img 

class ImageReader():
  def __init__(self, image_names, batch_size, threads):
    self.pool = ThreadPool(processes=threads)
    self.image_names = image_names
    self.batch_size = batch_size
    self.pos = 0

  def prefetch(self):
    if self.pos >= len(self.image_names):
      return False
    else:
      batch = self.image_names[self.pos: min(self.pos + self.batch_size, len(self.image_names))]
      self.pos += self.batch_size
      self.p = self.pool.map_async(get_image, batch)
      return True

  def get_next(self):
    if self.prefetch():
      res = self.p.get()
      res = np.float32(res)
      return res 
    else:
      print('Iterator exceed length')
      return None

def parse_fnames(fname):
  succ = []
  for i in open(fname):
    i = i.strip().split()[0]
    succ.append('./loose_crop_aligned/'+i)
  return succ


def calculateAcc(data):
  scores2=[]
  scores2.append(np.array(list(data)))
  model_names=[]
  model_names.append(pickles_saved)
  scores2=np.array(list(scores2))

  my_table=PrettyTable()
  x_labels=[10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1]
  my_table.field_names=['Model_names',str(x_labels[0]),str(x_labels[1]),str(x_labels[2]),str(x_labels[3]),str(x_labels[4]),str(x_labels[5]),str(x_labels[6])]

  plt.style.use('seaborn-darkgrid')
  sample_colors=['g','r','b','c','m','y','CadetBlue','Chocolate']
  for i,name in enumerate(model_names):
      print(scores2[0].shape)
      score=scores2[i][:,0]
      fpr,tpr,_=roc_curve(scores2[i][:,1],scores2[i][:,0])
      fpr=np.flipud(fpr)
      tpr=np.flipud(tpr)
      plt.plot(fpr,tpr,sample_colors[i],label=name)
      
      table_row=[]
      table_row.append(name)
      for fpr_iter in range(len(x_labels)):
          _, min_index = min(list(zip(abs(fpr-x_labels[fpr_iter]), range(len(fpr)))))
          table_row.append('%.4f' % tpr[min_index])
      my_table.add_row(table_row)
  plt.xlim(10**-7,0.1)
  plt.ylim(0.2,1)
  plt.grid(linestyle='--',linewidth=1)
  plt.xticks(x_labels)
  plt.yticks(np.linspace(0.2,1.0,8,endpoint=True))
  plt.xscale('log')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC on IJBC test2')
  print(my_table)


def cal_best_score(scores, labels):
  dt = list(zip(scores, labels))
  dt = sorted(dt, key=lambda x:x[0], reverse=True)
  corret = sum([1-i[1] for i in dt])
  best_num = corret
  best_thresh = np.array(dt)[0][0]

  for scr, lb in dt:
      if lb==1:
          corret += 1
      else: 
          corret -= 1
      if best_num<corret:
          best_num = corret
          best_thresh = scr 
  return  float(best_num)/len(scores)

def preprocess2(img,landmark=None):
  M = None
  image_size = [112,112]
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

    warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)

    return warped

def list_dir_file(gallery_dir):
  if os.name == 'nt':
    gallery_paths = glob.glob('.\\%s\\**'%gallery_dir,recursive=True)
  else:
    gallery_paths = glob.glob('./%s/**'%gallery_dir,recursive=True)
  return gallery_paths