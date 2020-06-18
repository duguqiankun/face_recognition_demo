import os
import random
import sys
import numpy as np
import cv2
from face_preprocess import preprocess
import glob
import time
sys.path.append('./face_detect/')
from detector import Retinaface_Detector
#from model import resnet18
from model_irse import IR_50
import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image


transform = transforms.Compose([
transforms.RandomHorizontalFlip(p=0.2),
transforms.ToTensor(),
transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

def drawing_box(image,box):
        #if box == None: return None
        xmin,ymin,xmax,ymax = box[0:4]
        xmin,ymin,xmax,ymax = int(xmin),int(ymin),int(xmax),int(ymax)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,102,0), 2)

def drawing_landmark(image,landmark):
        #if landmark == None: return None
        radius = 3
        color  = (0,255,0)
        thickness = -1
        for i in range(5):
            x,y = int(landmark[i]),int(landmark[5+i])
            cv2.circle(image,(x,y),radius,color, thickness)



def align_faces(frame,landmarks):
    landmarks=np.array(landmarks).astype(int)
    aligned_imgs=[]
    faces=landmarks.shape[0]
    for i in range(faces):
        lm=landmarks[i]
        print(frame.shape)
        img=preprocess(frame,lm)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = transform(Image.fromarray(img))
        aligned_imgs.append(img)
    aligned=torch.stack(aligned_imgs)
    return aligned


def drawResult(image,box,name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    xmin,ymin,xmax,ymax = box[0:4]
    xmin=int(xmin)
    ymin=int(ymin)
    xmax=int(xmax)
    ymax=int(ymax)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,102,0), 2)
    size_f=cv2.getTextSize(name,font,1,2)[0]
    cv2.rectangle(image, (xmin, ymin-size_f[1]-12), (xmin+size_f[0], ymin), (255,255,255),cv2.FILLED)
    cv2.putText(image,name,(xmin,ymin-10), font,1,(255,102,0),2,cv2.LINE_AA)


def getGalleryFeature(detector,fr_model):
    gallery_paths = glob.glob('./gallery/*')
    names=[]
    imgs_feats=[]
    for p in gallery_paths:
        if os.path.isfile(p):
            names.append(p.split('/')[len(p.split('/'))-1][:-4])
            img=cv2.imread(p)
            results=detector.detect(img)
            landmark=results[0][1]
            img=preprocess(img,landmark)

            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            img = transform(Image.fromarray(img)).unsqueeze(0).cuda()

            feat=fr_model(img)
            imgs_feats.append(feat)
    return names,imgs_feats




class FR_detector(object):
    """docstring for fr_detector"""
    def __init__(self,model,names,feats1):
        super(FR_detector, self).__init__()
        
        self.model = model
        self.names=names
        self.feats1= feats1.cpu().detach().numpy()



    def fr_detect(self,frame,landmarks):

        aligned_imgs=align_faces(frame,landmarks)
        out_names=[]
        out_scores=[]
        if len(aligned_imgs)>0:
            imgs=aligned_imgs.cuda()
            feats2=self.model(imgs).cpu().detach().numpy()
            sim=np.dot(self.feats1,feats2.T)
            result=np.argmax(sim, axis=0)
            score=np.max(sim, axis=0)
            for i in range(result.shape[0]):
                out_names.append(self.names[result[i]])
                out_scores.append(score[i])
        return out_names,out_scores



if __name__ == '__main__':

    detector=Retinaface_Detector(use_gpu=True)

    # fr_model = resnet18()
    # fr_model.load_state_dict(torch.load('./face_recognition/r18_arcface/model_r18_.pth.tar'))
    fr_model = IR_50([112,112])
    fr_model.load_state_dict(torch.load('./face_recognition/res50/backbone_ir50_ms1m.pth'))

    if torch.cuda.is_available():
        fr_model.cuda()
        print('GPU count: %d'%torch.cuda.device_count())
        print('CUDA is ready')
    fr_model.eval()
    names, feats1=getGalleryFeature(detector,fr_model)
    
    feats1=torch.stack(feats1).squeeze(1)
    print(feats1.shape)
    fr_detector=FR_detector(fr_model,names,feats1)




    cap = cv2.VideoCapture('IMG_0051.mp4')
    ret=True
    while(ret):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is not None:
            t0=time.time()


            boxes,landmarks= detector.detect2(frame)

            if len(landmarks)>0:
                names,scores=fr_detector.fr_detect(frame,landmarks)

                for i,box in enumerate(boxes):
                    if scores[i] >0.4:
                        drawResult(frame,box,names[i])

            cv2.imshow('frame',frame)
            print(time.time()-t0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    print('finish')
    cap.release()
    cv2.destroyAllWindows()