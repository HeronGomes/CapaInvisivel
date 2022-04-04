# -*- coding: utf-8 -*-
import cv2 as cv

import numpy as np

verdeMinimo = (60, 40, 35)
verdeMaximo = (100, 255, 255)

video = cv.VideoCapture(0)

bg = cv.imread('bg.jpg')
bg = cv.resize(bg,(400,400))

filme = cv.VideoWriter('./arte11.mp4',cv.VideoWriter.fourcc(*'mp4v'),15,(400,400))

while True:
    
    isCap,frame = video.read()
    
    if isCap:
        
        frame = cv.resize(frame,(400,400))
        imagemTabalho = frame.copy()
        
        blur = cv.GaussianBlur(imagemTabalho,(13,13),0)
        
        hsv = cv.cvtColor(blur,cv.COLOR_BGR2HSV)
        
        mask = cv.inRange(hsv, verdeMinimo,verdeMaximo)
        
        mask = cv.erode(mask,None,iterations=3)
        
        mask = cv.dilate(mask,None,iterations=3)
        
        # cv.imshow('Mascara',mask)
                        
        contours,_ = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        
        telaDesenho = np.zeros((400,400,3),np.uint8)
                       
        telaDesenho = cv.drawContours(telaDesenho,contours,-1,[255,255,255],cv.FILLED)
        
        
        pontosBrancos = np.where(telaDesenho==255)  
        
        telaDesenho[pontosBrancos] = bg[pontosBrancos]
        
        
        pontosBrancos = np.where(telaDesenho!=0)
        
        frame[pontosBrancos] = telaDesenho[pontosBrancos]
               
       
        cv.imshow('Stealth Camouflage 1.0',frame)
        filme.write(frame)

       
        tecla = cv.waitKey(100)
        
        if tecla == ord('q'):
            break

video.release()
filme.release()
cv.destroyAllWindows()


