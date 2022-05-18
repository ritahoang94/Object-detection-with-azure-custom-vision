import cv2
plantname = "pileadepressa"
vidcap = cv2.VideoCapture('input/videos/'+plantname+'.MOV')
count = 0

while(vidcap.isOpened()):

    success,image = vidcap.read()
    
    
    if success == True:
        count += 1
        # print(count)
        if count % 10 ==0:
            print("true")
            cv2.imwrite("input/images/"+plantname+'/'+plantname+ "_frame%d.jpg" % count, image)     # save frame as JPEG file      
      
        cv2.imshow("image",image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

vidcap.release()

cv2.destroyAllWindows() 

