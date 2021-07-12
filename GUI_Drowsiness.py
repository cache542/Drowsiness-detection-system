# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 21:58:30 2021

@author: ADITYA SINHA
"""

import numpy
from pygame import mixer
import time
import cv2
from tkinter import *
import tkinter.messagebox
import cv2
import os
from keras.models import load_model
import numpy as np
import requests
import json

def hel():
   tkinter.messagebox.showinfo("How to use","\n simply click on the buttons to use.\n For more details please contact the admin")

def Contri():
   tkinter.messagebox.showinfo("Contributors","\n1.Aditya Sinha\n2. Himanshu Attri\n3. Kunwar Dev \n4. Shivani \n")

def anotherWin():
   tkinter.messagebox.showinfo("About",'Drowsiness Detection version v1.0\n Made Using\n\t-OpenCV\n\t-CNN\n\t-Tkinter\nin Python , for detection of drowsiness of Driver')
   
def exitt():
   root.destroy()

def webrec():
   capture =cv2.VideoCapture(0)
   fourcc=cv2.VideoWriter_fourcc(*'XVID') 
   op=cv2.VideoWriter('Sample1.avi',fourcc,11.0,(640,480))
   while True:
      ret,frame=capture.read()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      cv2.imshow('frame',frame)
      op.write(frame)
      if cv2.waitKey(1) & 0xFF ==ord('q'):
         break
   op.release()
   capture.release()
   cv2.destroyAllWindows()  
def webdet():
    mixer.init()
    sound = mixer.Sound('beep-07.wav')
    
    face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
    leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
    reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')
    
    lbl=['Close','Open']
    model = load_model('models\cnncat2.h5')
    path = os.getcwd()
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    count=0
    score=0
    thicc=2
    rpred=[99]
    lpred=[99]
    
    fourcc=cv2.VideoWriter_fourcc(*'XVID') 
    op=cv2.VideoWriter('Sample1.avi',fourcc,11.0,(640,480))

    while(True):
        ret, frame = cap.read()
        height,width = frame.shape[:2] 


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
        left_eye = leye.detectMultiScale(gray)
        right_eye =  reye.detectMultiScale(gray)

        cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

        for (x,y,w,h) in right_eye:
            r_eye=frame[y:y+h,x:x+w]
            count=count+1
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye,(24,24))
            r_eye= r_eye/255
            r_eye=  r_eye.reshape(24,24,-1)
            r_eye = np.expand_dims(r_eye,axis=0)
            rpred = model.predict_classes(r_eye)
            if(rpred[0]==1):
                lbl='Open' 
            if(rpred[0]==0):
                lbl='Closed'
            break

        for (x,y,w,h) in left_eye:
            l_eye=frame[y:y+h,x:x+w]
            count=count+1
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(24,24,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            lpred = model.predict_classes(l_eye)
            if(lpred[0]==1):
                lbl='Open'   
            if(lpred[0]==0):
                lbl='Closed'
            break

        if(rpred[0]==0 and lpred[0]==0):
            score=score+1
            cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
        else:
            score=score-1
            cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
        if(score<0):
            score=0   
        cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        if(score>15):
        #person is feeling sleepy so we beep the alarm
            cv2.imwrite(os.path.join(path,'image.jpg'),frame)
            try:
                sound.play()
                #tkinter.messagebox.showinfo("city:",city)
            
            except:  # isplaying = False
                pass
            if(thicc<16):
                thicc= thicc+2
            else:
                thicc=thicc-2
                if(thicc<2):
                    thicc=2
            cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
        cv2.imshow('frame',frame)
        op.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    op.release()
    cap.release()
    cv2.destroyAllWindows()

root=Tk()
root.geometry('500x570')
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
root.title('Drowsiness Detection')
frame.config(background='sky blue')
label = Label(frame, text="Drowsiness Detection",bg='sky blue',font=('Times 35 bold'))
label.pack(side=TOP)

label1 = Label(root, bg = 'white')
label1.pack(side=TOP)

filename = PhotoImage(file="demo.png")
background_label = Label(frame,image=filename)
background_label.pack(side=TOP)

menu = Menu(root)
root.config(menu=menu)

subm1 = Menu(menu)
menu.add_cascade(label="Tools",menu=subm1)
subm1.add_command(label="Help",command=hel)

subm2 = Menu(menu)
menu.add_cascade(label="About",menu=subm2)
subm2.add_command(label="Project",command=anotherWin)
subm2.add_command(label="Contributors",command=Contri)


but1=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=webrec,text='Open Camera & Record',font=('helvetica 15 bold'))
but1.place(x=5,y=174)

but2=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=webdet,text='Open Camera & Detect',font=('helvetica 15 bold'))
but2.place(x=5,y=250)

#but3=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=loc,text='Know your location',font=('helvetica 15 bold'))
#but3.place(x=5,y=325)
but4=Button(frame,padx=5,pady=5,width=5,bg='white',fg='black',relief=GROOVE,text='EXIT',command=exitt,font=('helvetica 15 bold'))
but4.place(x=210,y=478)


root.mainloop()