#pip install opencv-contrib-python
import cv2 as cv
cam = cv.VideoCapture(0)#for linux user put the value to 1 instead of 0
cc = cv.VideoWriter_fourcc(*'mp4v')
file = cv.VideoWriter('dataset/Training/output.mp4', cc, 15.0, (640, 480))
if not cam.isOpened():
   print("error opening camera")
   exit()
while True:
   # Capture frame-by-frame
   ret, frame = cam.read()
   frame = cv.flip(frame, 1)
   # if frame is read correctly ret is True
   if not ret:
      print("error in retrieving frame")
      break
   img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
   cv.imshow('frame', img)
   file.write(img)

   
   if cv.waitKey(1) == ord('q'):
      break

cam.release()
file.release()
cv.destroyAllWindows()