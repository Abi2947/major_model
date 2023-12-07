import cv2
import numpy as np
import os
import string
# Create the directory structure
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/train"):
    os.makedirs("data/train")
if not os.path.exists("data/test"):
    os.makedirs("data/test")
for i in range(3):
    if not os.path.exists("data/train/" + str(i)):
        os.makedirs("data/train/"+str(i))
    if not os.path.exists("data/test/" + str(i)):
        os.makedirs("data/test/"+str(i))

for i in string.ascii_uppercase:
    if not os.path.exists("data/train/" + i):
        os.makedirs("data/train/"+i)
    if not os.path.exists("data/test/" + i):
        os.makedirs("data/test/"+i)
    


# Train or test 
mode = 'train'
directory = 'data/'+mode+'/'
minValue = 70

cap = cv2.VideoCapture(0)
interrupt = -1  

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Getting count of existing images
    count = {
            '१':len(os.listdir(directory+"/१")),
            '२':len(os.listdir(directory+"/२")),
            '३':len(os.listdir(directory+"/३")),
            '४':len(os.listdir(directory+"/४")),
            '५':len(os.listdir(directory+"/५")),
            '६':len(os.listdir(directory+"/६")),
            '७':len(os.listdir(directory+"/७")),
            '८':len(os.listdir(directory+"/८")),
            '९':len(os.listdir(directory+"/९")),
            '०':len(os.listdir(directory+"/०")),
            ''''आ': len(os.listdir(directory+"/आ")),
            'इ': len(os.listdir(directory+"/इ")),
            'ई': len(os.listdir(directory+"/ई")),
            'उ': len(os.listdir(directory+"/उ")),
            'ऊ': len(os.listdir(directory+"/ऊ")),
            'ऋ': len(os.listdir(directory+"/ऋ")),
            'ए': len(os.listdir(directory+"/ए")),
            'ऐ': len(os.listdir(directory+"/ऐ")),
            'ओ': len(os.listdir(directory+"/ओ")),
            'औ': len(os.listdir(directory+"/औ")),
            'अं': len(os.listdir(directory+"/अं")),
            'अः': len(os.listdir(directory+"/अः")),'''
             'क': len(os.listdir(directory+"/क")),
             'ख': len(os.listdir(directory+"/ख")),
             'ग': len(os.listdir(directory+"/ग")),
             'घ': len(os.listdir(directory+"/घ")),
             'ङ': len(os.listdir(directory+"/ङ")),
             'च': len(os.listdir(directory+"/च")),
             'छ': len(os.listdir(directory+"/छ")),
             'ज': len(os.listdir(directory+"/ज")),
             'झ': len(os.listdir(directory+"/झ")),
             'ञ': len(os.listdir(directory+"/ञ")),
             'ट': len(os.listdir(directory+"/ट")),
             'ठ': len(os.listdir(directory+"/ठ")),
             'ड': len(os.listdir(directory+"/ड")),
             'ढ': len(os.listdir(directory+"/ढ")),
             'ण': len(os.listdir(directory+"/ण")),
             'त': len(os.listdir(directory+"/त")),
             'थ': len(os.listdir(directory+"/थ")),
             'द': len(os.listdir(directory+"/द")),
             'ध': len(os.listdir(directory+"/ध")),
             'न': len(os.listdir(directory+"/न")),
             'प': len(os.listdir(directory+"/प")),
             'फ': len(os.listdir(directory+"/फ")),
             'ब': len(os.listdir(directory+"/ब")),
             'भ': len(os.listdir(directory+"/भ")),
             'म': len(os.listdir(directory+"/म")),
             'य': len(os.listdir(directory+"/य")),
             'र': len(os.listdir(directory+"/र")),
             'ल': len(os.listdir(directory+"/ल")),
             'व': len(os.listdir(directory+"/व")),
             'श': len(os.listdir(directory+"/श")),
             'ष': len(os.listdir(directory+"/ष")),
             'स': len(os.listdir(directory+"/स")),
             'ह': len(os.listdir(directory+"/ह")),
             'क्ष': len(os.listdir(directory+"/क्ष")),
             'त्र': len(os.listdir(directory+"/त्र")),
             'ज्ञ': len(os.listdir(directory+"/ज्ञ"))
             }
    
    # Printing the count in each set to the screen
    cv2.putText(frame, "० : "+str(count['०']), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "१ : "+str(count['१']), (10, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "२ : "+str(count['२']), (10, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "३ : "+str(count['३']), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "४ : "+str(count['४']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "५ : "+str(count['५']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "६ : "+str(count['६']), (10, 130), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "७ : "+str(count['७']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "८ : "+str(count['८']), (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "९ : "+str(count['९']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "क : "+str(count['क']), (10, 170), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ख : "+str(count['ख']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ग : "+str(count['ग']), (10, 190), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "घ : "+str(count['घ']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ङ : "+str(count['ङ']), (10, 210), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "च : "+str(count['च']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "छ : "+str(count['छ']), (10, 230), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ज : "+str(count['ज']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "झ : "+str(count['झ']), (10, 250), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ञ : "+str(count['ञ']), (10, 260), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ट : "+str(count['ट']), (10, 270), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ठ : "+str(count['ठ']), (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ड : "+str(count['ड']), (10, 290), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ढ : "+str(count['ढ']), (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ण : "+str(count['ण']), (10, 310), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "त : "+str(count['त']), (10, 320), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "थ : "+str(count['थ']), (10, 330), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "द : "+str(count['द']), (10, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ध : "+str(count['ध']), (10, 350), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "न : "+str(count['न']), (10, 360), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "प : "+str(count['प']), (10, 370), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "फ : "+str(count['फ']), (10, 380), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ब : "+str(count['ब']), (10, 390), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "भ : "+str(count['भ']), (10, 400), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "म : "+str(count['म']), (10, 410), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "य : "+str(count['य']), (10, 420), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "र : "+str(count['र']), (10, 430), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ल : "+str(count['ल']), (10, 440), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "व : "+str(count['व']), (10, 450), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "श : "+str(count['श']), (10, 460), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ष : "+str(count['ष']), (10, 470), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "स : "+str(count['स']), (10, 480), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ह : "+str(count['ह']), (10, 490), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "क्ष : "+str(count['क्ष']), (10, 500), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "त्र : "+str(count['त्र']), (10, 510), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ज्ञ : "+str(count['ज्ञ']), (10, 520), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)


    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (220-1, 9), (620+1, 419), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[10:410, 220:520]
#    roi = cv2.resize(roi, (64, 64))
    
    cv2.imshow("Frame", frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray,(5,5),2)
    # #blur = cv2.bilateralFilter(roi,9,75,75)
    
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #time.sleep(5)
    #cv2.imwrite("/home/rc/Downloads/soe/im1.jpg", roi)
    #test_image = func("/home/rc/Downloads/soe/im1.jpg")


    
    test_image = cv2.resize(test_image, (300,300))
    cv2.imshow("test", test_image)
        
    #_, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    #kernel = np.ones((1, 1), np.uint8)
    #img = cv2.dilate(mask, kernel, iterations=1)
    #img = cv2.erode(mask, kernel, iterations=1)
    # do the processing after capturing the image!
#    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#    _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    
    ##gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ##cv2.imshow("GrayScale", gray)
    ##blur = cv2.GaussianBlur(gray,(5,5),2)
    
    #blur = cv2.bilateralFilter(roi,9,75,75)
    
    ##th3 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ##ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # cv2.imshow("ROI", roi)
    #roi = frame
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    if interrupt & 0xFF == ord('०'):
        cv2.imwrite(directory+'०/'+str(count['०'])+'.jpg', roi)
    if interrupt & 0xFF == ord('१'):
        cv2.imwrite(directory+'१/'+str(count['१'])+'.jpg', roi)
    if interrupt & 0xFF == ord('२'):
        cv2.imwrite(directory+'२/'+str(count['२'])+'.jpg', roi)
    if interrupt & 0xFF == ord('३'):
        cv2.imwrite(directory+'३/'+str(count['३'])+'.jpg', roi)
    if interrupt & 0xFF == ord('४'):
        cv2.imwrite(directory+'४/'+str(count['४'])+'.jpg', roi) 
    if interrupt & 0xFF == ord('५'):
        cv2.imwrite(directory+'५/'+str(count['५'])+'.jpg', roi) 
    if interrupt & 0xFF == ord('६'):
        cv2.imwrite(directory+'६/'+str(count['६'])+'.jpg', roi) 
    if interrupt & 0xFF == ord('७'):
        cv2.imwrite(directory+'७/'+str(count['७'])+'.jpg', roi) 
    if interrupt & 0xFF == ord('८'):
        cv2.imwrite(directory+'८/'+str(count['८'])+'.jpg', roi) 
    if interrupt & 0xFF == ord('९'):
        cv2.imwrite(directory+'९/'+str(count['९'])+'.jpg', roi) 
    if interrupt & 0xFF == ord('क'):
        cv2.imwrite(directory+'क/'+str(count['क'])+'.jpg', roi)
    if interrupt & 0xFF == ord('ख'):
        cv2.imwrite(directory+'ख/'+str(count['ख'])+'.jpg', roi)
    if interrupt & 0xFF == ord('ग'):
        cv2.imwrite(directory+'ग/'+str(count['ग'])+'.jpg', roi)
    if interrupt & 0xFF == ord('घ'):
        cv2.imwrite(directory+'घ/'+str(count['घ'])+'.jpg', roi)
    if interrupt & 0xFF == ord('ङ'):
        cv2.imwrite(directory+'ङ/'+str(count['ङ'])+'.jpg', roi)
    if interrupt & 0xFF == ord('च'):
        cv2.imwrite(directory+'च/'+str(count['च'])+'.jpg', roi)
    if interrupt & 0xFF == ord('छ'):
        cv2.imwrite(directory+'छ/'+str(count['छ'])+'.jpg', roi)
    if interrupt & 0xFF == ord('ज'):
        cv2.imwrite(directory+'ज/'+str(count['ज'])+'.jpg', roi)
    if interrupt & 0xFF == ord('झ'):
        cv2.imwrite(directory+'झ/'+str(count['झ'])+'.jpg', roi)
    if interrupt & 0xFF == ord('ञ'):
        cv2.imwrite(directory+'ञ/'+str(count['ञ'])+'.jpg', roi)
    if interrupt & 0xFF == ord('ट'):
        cv2.imwrite(directory+'ट/'+str(count['ट'])+'.jpg', roi)
    if interrupt & 0xFF == ord('ठ'):
        cv2.imwrite(directory+'ठ/'+str(count['ठ'])+'.jpg', roi)
    if interrupt & 0xFF == ord('ड'):
        cv2.imwrite(directory+'ड/'+str(count['ड'])+'.jpg', roi)
    if interrupt & 0xFF == ord('ढ'):
        cv2.imwrite(directory+'ढ/'+str(count['ढ'])+'.jpg', roi)
    if interrupt & 0xFF == ord('ण'):
        cv2.imwrite(directory+'ण/'+str(count['ण'])+'.jpg', roi)
    if interrupt & 0xFF == ord('त'):
        cv2.imwrite(directory+'त/'+str(count['त'])+'.jpg', roi)
    if interrupt & 0xFF == ord('थ'):
        cv2.imwrite(directory+'थ/'+str(count['थ'])+'.jpg', roi)
    if interrupt & 0xFF == ord('द'):
        cv2.imwrite(directory+'द/'+str(count['द'])+'.jpg', roi)
    if interrupt & 0xFF == ord('ध'):
        cv2.imwrite(directory+'ध/'+str(count['ध'])+'.jpg', roi)
    if interrupt & 0xFF == ord('न'):
        cv2.imwrite(directory+'न/'+str(count['न'])+'.jpg', roi)
    if interrupt & 0xFF == ord('प'):
        cv2.imwrite(directory+'प/'+str(count['प'])+'.jpg', roi)
    if interrupt & 0xFF == ord('फ'):
        cv2.imwrite(directory+'फ/'+str(count['फ'])+'.jpg', roi)
    if interrupt & 0xFF == ord('ब'):
        cv2.imwrite(directory+'ब/'+str(count['ब'])+'.jpg', roi)
    if interrupt & 0xFF == ord('भ'):
        cv2.imwrite(directory+'भ/'+str(count['भ'])+'.jpg', roi)
    if interrupt & 0xFF == ord('म'):
        cv2.imwrite(directory+'म/'+str(count['म'])+'.jpg', roi)
    if interrupt & 0xFF == ord('य'):
        cv2.imwrite(directory+'य/'+str(count['य'])+'.jpg', roi)        
    if interrupt & 0xFF == ord('र'):
        cv2.imwrite(directory+'र/'+str(count['र'])+'.jpg', roi)        
    if interrupt & 0xFF == ord('ल'):
        cv2.imwrite(directory+'ल/'+str(count['ल'])+'.jpg', roi)        
    if interrupt & 0xFF == ord('व'):
        cv2.imwrite(directory+'व/'+str(count['व'])+'.jpg', roi)        
    if interrupt & 0xFF == ord('श'):
        cv2.imwrite(directory+'श/'+str(count['श'])+'.jpg', roi)        
    if interrupt & 0xFF == ord('ष'):
        cv2.imwrite(directory+'ष/'+str(count['ह'])+'.jpg', roi)        
    if interrupt & 0xFF == ord('स'):
        cv2.imwrite(directory+'स/'+str(count['स'])+'.jpg', roi)        
    if interrupt & 0xFF == ord('ह'):
        cv2.imwrite(directory+'ह/'+str(count['ह'])+'.jpg', roi)        
    if interrupt & 0xFF == ord('क्ष'):
        cv2.imwrite(directory+'क्ष/'+str(count['क्ष'])+'.jpg', roi)        
    if interrupt & 0xFF == ord('त्र'):
        cv2.imwrite(directory+'त्र/'+str(count['त्र'])+'.jpg', roi)
    if interrupt & 0xFF == ord('ज्ञ'):
        cv2.imwrite(directory+'ज्ञ/'+str(count['ज्ञ'])+'.jpg', roi) 
cap.release()
cv2.destroyAllWindows()
"""
d = "old-data/test/0"
newd = "data/test/0"
for walk in os.walk(d):
    for file in walk[2]:
        roi = cv2.imread(d+"/"+file)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        cv2.imwrite(newd+"/"+file, mask)     
"""