import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

pTime=0
wScr,hScr = autopy.screen.size()
#print(wScr,hScr) 
# 1366,768
wCam, hCam = 640,480
frameR=80 #frameReduction
smoothening=5
plocX,plocY=0,0
clocX,clocY=0,0


cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

detector = htm.handDetector(maxHands=1)
while True:
    # 1. Find hand Landmarks
    success,img = cap.read()
    img = detector.findHands(img)
    lmList,bbox=detector.findPosition(img)
    #print(lmList)
    try:
        # 2. Get the tip of the index and middle fingers
        if len(lmList)!=0:
            x1,y1=lmList[8][1:]
            x2,y2=lmList[12][1:]
            #print(x1,y1,x2,y2)
            
            # 3. check which fingers are up
            fingers=detector.fingersUp()
            cv2.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(182,144,69),2) #inner rectangle
            # print(fingers)
            
            # 4. only Index Fingers : Moving Mode
            if fingers[1]==1 and fingers[2]==0:
            # 5. Convert Coordinates
                x3=np.interp(x1,(frameR,wCam-frameR),(0,wScr))
                y3=np.interp(y1,(frameR,hCam-frameR),(0,wScr))
                
                # 6. Smoothen values
                clocX=plocX+(x3-plocX)/smoothening
                clocY=plocY+(y3-plocY)/smoothening
                
                # 7. Move Mouse
                autopy.mouse.move(wScr-clocX,clocY)
                cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
                plocX,plocY=clocX,clocY     
            
            # 8. Both Index and middle fingers are up: clickng mouse
            if fingers[1]==1 and fingers[2]==1:
                cv2.circle(img,(x1,y1),5,(255,0,255),cv2.FILLED)
                cv2.circle(img,(x2,y2),5,(255,0,255),cv2.FILLED)
                # 9. Find distance between fingers
                length, img, lineinfo = detector.findDistance(8,12,img)
                #print(length)
                
                # 10.Click mouse if distance short
                if length<35:
                    cv2.circle(img,(lineinfo[4],lineinfo[5]),10,(0,255,0),cv2.FILLED)
                    autopy.mouse.click()

    except:
        pass
    
    
    
    
    
    
    
    
    # 11. Frame Rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    
    # 12.Display
    cv2.imshow("Image",img)
    cv2.waitKey(1)