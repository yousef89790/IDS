from tkinter import *
from ids.KDDCUP99 import *
from ids.UNSW_NB15_EDA.UNSW_NB15_EDA import *
from attacks.DOSS import *
from attacks.R2l.R2L import *
from ScanVirus.Scan_v2 import *
from attacks.Keylogger.Keylogger import *
from tkinter.filedialog import askdirectory
from tkinter import messagebox

import os
import hashlib

import pandas as pd

# Read the data

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
# visualization for KPI
import time

# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

import threading


from PIL import ImageTk, Image


#---------------------------------------------------- DeshBoard Panel -----------------------------------------


def home():

	# --------------------Tkinter Base Setup ---------------------#

    window = Tk()
    window.title("Protect your self")

    window.update_idletasks()
    height_m = 750
    width_m = 1200
    # height_m = window.winfo_reqheight()
    # width_m = window.winfo_reqwidth()
    x_m = (window.winfo_screenwidth()//2)-(width_m//2)
    y_m = (window.winfo_screenheight()//2)-(height_m//2)
    window.geometry('{}x{}+{}+{}'.format(width_m, height_m, x_m, y_m))

    # def ExitWindow():
    #     window.quit()

    # window.protocol("WM_DELETE_WINDOW", ExitWindow)

    winFrame = Frame(window, width="1200", height="750", bg="black")
    winFrame.pack()
    winFrame.pack_propagate(0)


    

    def ExitWindow():
        window.quit()

    window.protocol("WM_DELETE_WINDOW", ExitWindow)

    # --------------------Tkinter Base Setup End ------------------#


    # --------------------Global Var --------------------#
    # global winFrame

    # --------------------Global Var End -----------------#

    winFrame.destroy()

    # --------------------Functions------------------#

    # Define function to hide the widget

    def hide_widget(widget):
        widget.pack_forget()
        widget.place_forget()
        widget.grid_forget()

    # Define a function to show the widget
    def show_widget(widget,X,Y):
        # widget.pack()
        widget.place(x=X, y=Y)

    def fnIDS(event):
        hide_widget(dossButton)
        hide_widget(r2lButton)
        hide_widget(keyloggerButton)
        show_widget(kddButton,155,400)
        show_widget(unswButton,400,400)

    def fnAttacks(event):
        hide_widget(kddButton)
        hide_widget(unswButton)
        show_widget(dossButton,155,400)
        show_widget(r2lButton,400,400)
        show_widget(keyloggerButton,645,400)

    

    # --------------------End KDD Function------------------#

    #--------------------Main Frame ---------------------#

    winFrame = Frame(window, width="1200",height="750",bg="white")
    winFrame.pack()
    winFrame.pack_propagate(0)

    # txtFrame = Frame(winFrame, width="1200",height="450",bg="white")
    # txtFrame.pack()
    # txtFrame.pack_propagate(0)


    backgroundImage = PhotoImage(file="res\\home.png", master=winFrame)
    bg_image = Label(
        winFrame,
        image=backgroundImage
    )
    bg_image.pack()

    #--------------------Main Frame End ------------------#

    # --------------------KDDCUP99 Button --------------------#

    kddButtonImg = PhotoImage(file="res\\Buttons\\Non-Hoved\\KDDCUP99.png")
    hovKddButtonImg = PhotoImage(file="res\\Buttons\\Hoved\\hovKDDCUP99.png")


    def KddButtonEnterFrame(event):
        kddButton.config(image=hovKddButtonImg)

    def KddButtonLeaveFrame(event):
        kddButton.config(image=kddButtonImg)


    kddButton = Label(winFrame, image=kddButtonImg,bg="white",cursor="hand2")
    kddButton.place(x=155, y=400)

    kddButton.bind('<Enter>', KddButtonEnterFrame)
    kddButton.bind('<Leave>', KddButtonLeaveFrame)
    # kddButton.bind("<Button-1>",get_KDDCUP99)
    # kddButton.bind("<Button-1>",lambda e: get_KDDCUP99_thread)
    kddButton.bind("<Button-1>", lambda e:threading.Thread(target=KDDCUP99).start())

    # --------------------KDDCUP99 Button End------------------#

    # --------------------UNSW-NB15-EDA Button --------------------#

    unswButtonImg = PhotoImage(
        file="res\\Buttons\\Non-Hoved\\UNSW-NB15-EDA.png")
    hovUnswButtonImg = PhotoImage(
        file="res\\Buttons\\Hoved\\hovUNSW-NB15-EDA.png")


    def UnswButtonEnterFrame(event):
        unswButton.config(image=hovUnswButtonImg)

    def UnswButtonLeaveFrame(event):
        unswButton.config(image=unswButtonImg)


    unswButton = Label(winFrame, image=unswButtonImg,bg="white",cursor="hand2")
    unswButton.place(x=400, y=400)

    unswButton.bind('<Enter>', UnswButtonEnterFrame)
    unswButton.bind('<Leave>', UnswButtonLeaveFrame)
    unswButton.bind("<Button-1>", lambda e:threading.Thread(target=UNSW_NB15_EDA).start())

    # --------------------UNSW-NB15-EDA Button End------------------#


# --------------------IDS Button --------------------#

    idsButtonImg = PhotoImage(file="res\\Buttons\\Non-Hoved\\ids.png")
    hovIdsButtonImg = PhotoImage(file="res\\Buttons\\Hoved\\hovIds.png")


    def IdsButtonEnterFrame(event):
        idsButton.config(image=hovIdsButtonImg)

    def IdsButtonLeaveFrame(event):
        idsButton.config(image=idsButtonImg)


    idsButton = Label(winFrame, image=idsButtonImg,bg="white",cursor="hand2")
    idsButton.place(x=155, y=500)

    idsButton.bind('<Enter>', IdsButtonEnterFrame)
    idsButton.bind('<Leave>', IdsButtonLeaveFrame)
    idsButton.bind("<Button-1>", fnIDS)

    # --------------------IDS Button End------------------#

    # --------------------DOSS Button --------------------#

    dossButtonImg = PhotoImage(file="res\\Buttons\\Non-Hoved\\doss.png")
    hovDossButtonImg = PhotoImage(file="res\\Buttons\\Hoved\\hovDoss.png")


    def DossButtonEnterFrame(event):
        dossButton.config(image=hovDossButtonImg)

    def DossButtonLeaveFrame(event):
        dossButton.config(image=dossButtonImg)

    # def threading():
    #     # Call work function
    #     t1=threading.Thread(target=get_KDDCUP99)
    #     t1.start()


    dossButton = Label(winFrame, image=dossButtonImg,bg="white",cursor="hand2")
    dossButton.place(x=155, y=400)

    dossButton.bind('<Enter>', DossButtonEnterFrame)
    dossButton.bind('<Leave>', DossButtonLeaveFrame)
    # dossButton.bind("<Button-1>",get_KDDCUP99)
    # dossButton.bind("<Button-1>",lambda e: get_KDDCUP99_thread)
    dossButton.bind("<Button-1>", lambda e:threading.Thread(target=DOSS).start())

    # --------------------Doss Button End------------------#

    # --------------------R2L Button --------------------#

    r2lButtonImg = PhotoImage(
        file="res\\Buttons\\Non-Hoved\\r2l.png")
    hovR2lButtonImg = PhotoImage(
        file="res\\Buttons\\Hoved\\hovR2l.png")


    def R2lButtonEnterFrame(event):
        r2lButton.config(image=hovR2lButtonImg)

    def R2lButtonLeaveFrame(event):
        r2lButton.config(image=r2lButtonImg)


    r2lButton = Label(winFrame, image=r2lButtonImg,bg="white",cursor="hand2")
    r2lButton.place(x=400, y=400)

    r2lButton.bind('<Enter>', R2lButtonEnterFrame)
    r2lButton.bind('<Leave>', R2lButtonLeaveFrame)
    r2lButton.bind("<Button-1>", lambda e:threading.Thread(target=R2L).start())

    # --------------------R2l Button End------------------#

    # --------------------Keylogger Button --------------------#

    keyloggerButtonImg = PhotoImage(
        file="res\\Buttons\\Non-Hoved\\keylogger.png")
    hovKeyloggerButtonImg = PhotoImage(
        file="res\\Buttons\\Hoved\\hovKeylogger.png")


    def KeyloggerButtonEnterFrame(event):
        keyloggerButton.config(image=hovKeyloggerButtonImg)

    def KeyloggerButtonLeaveFrame(event):
        keyloggerButton.config(image=keyloggerButtonImg)


    keyloggerButton = Label(winFrame, image=keyloggerButtonImg,bg="white",cursor="hand2")
    keyloggerButton.place(x=645, y=400)

    keyloggerButton.bind('<Enter>', KeyloggerButtonEnterFrame)
    keyloggerButton.bind('<Leave>', KeyloggerButtonLeaveFrame)
    keyloggerButton.bind("<Button-1>", lambda e:threading.Thread(target=Keylogger).start())

    # --------------------keylogger Button End------------------#


    # --------------------Attacks Button -------------------#

    systemButtonImg = PhotoImage(file="res\\Buttons\\Non-Hoved\\attacks.png")
    hovSystemButtonImg = PhotoImage(file="res\\Buttons\\Hoved\\hovAttacks.png")

    def SystemButtonEnterFrame(event):
        systemButton.config(image=hovSystemButtonImg)

    def SystemButtonLeaveFrame(event):
        systemButton.config(image=systemButtonImg)


    systemButton = Label(winFrame, image=systemButtonImg,bg="white",cursor="hand2")
    systemButton.place(x=335, y=500)

    systemButton.bind('<Enter>', SystemButtonEnterFrame)
    systemButton.bind('<Leave>', SystemButtonLeaveFrame)
    systemButton.bind("<Button-1>", fnAttacks)

    # --------------------Attacks Button End ---------------#

    # --------------------Scan Button ---------------------#

    # def close():
    #     window.destroy()	
            
    # def scan_now():
    #         try:
    #             close()
    #             Scan_v2()
    #         except Exception as es:
    #             messagebox.showerror("Error" , f"Error Dui to : {str(es)}", parent = window)



    scanButtonImg = PhotoImage(file="res\\Buttons\\Non-Hoved\\scan.png")
    hovScanButtonImg = PhotoImage(file="res\\Buttons\\Hoved\\hovScan.png")

    def ScanButtonEnterFrame(event):
        scanButton.config(image=hovScanButtonImg)

    def ScanButtonLeaveFrame(event):
        scanButton.config(image=scanButtonImg)


    scanButton = Label(winFrame, image=scanButtonImg,bg="white",cursor="hand2")
    scanButton.place(x=515, y=500)
    # scanButton = Button(winFrame, text = "Scan" ,font='Verdana 10 bold',command = scan_now)
    # scanButton.place(x=515, y=500)

    #(x=515, y=500)
    

    scanButton.bind('<Enter>', ScanButtonEnterFrame)
    scanButton.bind('<Leave>', ScanButtonLeaveFrame)
    # scanButton.bind("<Button-1>", lambda e:Scan_v2)
    scanButton.bind("<Button-1>", lambda e:threading.Thread(target=Scan_v2).start())


    # --------------------Scan Button End------------------#

    


    # --------------------Accounts Button -----------------#

    webButtonImg = PhotoImage(file="res\\Buttons\\Non-Hoved\\web.png")
    hovWebButtonImg = PhotoImage(file="res\\Buttons\\Hoved\\hovWeb.png")

    def WebButtonEnterFrame(event):
        webButton.config(image=hovWebButtonImg)

    def WebButtonLeaveFrame(event):
        webButton.config(image=webButtonImg)


    webButton = Label(winFrame, image=webButtonImg,bg="white",cursor="hand2")
    webButton.place(x=695, y=500)

    webButton.bind('<Enter>', WebButtonEnterFrame)
    webButton.bind('<Leave>', WebButtonLeaveFrame)

    # --------------------Accounts Button End -------------#

    # --------------------Internet Button -----------------#

    toolsButtonImg = PhotoImage(file="res\\Buttons\\Non-Hoved\\tools.png")
    hovToolsButtonImg = PhotoImage(file="res\\Buttons\\Hoved\\hovTools.png")

    def ToolsButtonEnterFrame(event):
        toolsButton.config(image=hovToolsButtonImg)

    def ToolsButtonLeaveFrame(event):
        toolsButton.config(image=toolsButtonImg)


    toolsButton = Label(winFrame, image=toolsButtonImg,bg="white",cursor="hand2")
    toolsButton.place(x=875, y=500)

    toolsButton.bind('<Enter>', ToolsButtonEnterFrame)
    toolsButton.bind('<Leave>', ToolsButtonLeaveFrame)

    # --------------------Internet Button End -------------#


    # --------------------Footer         ---- -------------#

    # global footerBannerImg
    # footerBannerImg = PhotoImage(file="res\\footer.png")

    # footerBanner = Label(winFrame,image=footerBannerImg,bg="white")
    # footerBanner.place(x=300,y=700)

    # --------------------Footer End--------------#

    window.mainloop()

					
#-----------------------------------------------------End Deshboard Panel -------------------------------------