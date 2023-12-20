import cv2
import os

cam = cv2.VideoCapture("dataset/Training/output.mp4")

d_path=r'D:\Major Project\dataset\Training'
os.chdir(d_path)
frameno = 0
while(True):
   ret,frame = cam.read()
   if ret:
      # if video is still left continue creating images
      name = str(frameno) + '.jpg'
      print ('new frame captured...' + name)

      cv2.imwrite(name, frame)
      frameno += 1
   else:
      break

cam.release()
cv2.destroyAllWindows()