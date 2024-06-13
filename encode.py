import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{'databaseURL':"https://faceface-f5d7f-default-rtdb.firebaseio.com/",
                                    'storageBucket':'faceface-f5d7f.appspot.com'
                                    })

#load img student
flodersPath="Images"
PathList=os.listdir(flodersPath)
imgList=[] # convort to number
studentIds=[]
for path in PathList:
    imgList.append(cv2.imread(os.path.join(flodersPath,path)))
    studentIds.append(os.path.splitext(path)[0])
    fileName=f'{flodersPath}/{path}'
    bucket=storage.bucket()
    blob=bucket.blob(fileName)
    blob.upload_from_filename(fileName)
print(studentIds)
def findencodings(imgs):
    encodelist=[]
    for img in imgs:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist
encodelistknew=findencodings(imgList)
encodelistknewwithids=[encodelistknew,studentIds]

file=open('Encoding.p',"wb")
pickle.dump(encodelistknewwithids,file)
file.close()
