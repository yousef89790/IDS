from tkinter import *
from ids.KDDCUP99 import *
from ids.UNSW_NB15_EDA.UNSW_NB15_EDA import *
from attacks.DOSS import *
from attacks.R2l.R2L import *
from ScanVirus.Scan import *
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

    window.geometry("1200x750")
    window.minsize("1200", "750")
    window.maxsize("1200", "750")

    winFrame = Frame(window, width="1200", height="750", bg="black")
    winFrame.pack()
    winFrame.pack_propagate(0)

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

    def get_input(event):
        text.insert(END, "Welcome to Terminal..."+"\n")
        text.insert(END, "Welcome to End..."+"\n")
        text.see(END)

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

    def insert_text(txt):
        text.insert(END, txt+"\n")
        text.see(END)

    # def scan():
    #     file_list = []
    #     root_dir = r'{}'.format(askdirectory(title="Choose Folder",initialdir=r'D:\\Projects\\Antivirus\\Antivirus\\ScanFolder\\'))
    #     insert_text(root_dir)
    #     insert_text("Start Scan ............")
    #     for subdir,dirs,files in os.walk(root_dir):
    #         for file in files:

    #             file_path=subdir + os.sep + file
                
    #             if(file_path.endswith(".exe") or file_path.endswith(".dll")):
    #                 file_list.append(file_path)
        
    #     print(" we found some files could be viruses")
    #     insert_text(" we found some files could be viruses")
    #     print("start scan files....")
    #     insert_text("start scan files....")

    #     #  sleep for 10 second
    #     def counts():
    #         for x in range(5):
    #             print(x+1)
    #             insert_text(str(x+1))
    #             time.sleep(1)
    #     counts()

    #     def loading():
    #         path = 'res\\Scan.gif'

    #         # root = Tk()
    #         # img = ImageTk.PhotoImage(Image.open(path))
    #         # panel = Label(root, image = img)
    #         # panel.pack(side = "bottom", fill = "both", expand = "yes")
    #         # root.mainloop()

    #         splash_root = Tk()
    #         splash_root.title("Splash Screen")
    #         splash_root.geometry("300x200")
    #         time.sleep(5)
    #     loading()
		
    #     # ملفات معروفة انها فيروس من النت
    #     def scan():
    #         #ملفات متفيرسة
    #         infected_list=[]

    #         for f in file_list:

    #             virus_def=open("ScanVirus\\viruses.txt","r")
    #             file_not_read=False

    #             print("\n scaning... : {}".format(f))
    #             text.insert(END, "Scaning... : {}".format(f))
    #             text.see(END)
    #             hasher=hashlib.md5()

    #             try:
    #                 with open(f,"rb") as file:
    #                     try:

    #                         buf=file.read()
    #                         file_not_read=True

    #                         hasher.update(buf)

    #                         file_hashed=hasher.hexdigest()
    #                         print("file md5 Done:{}".format(file_hashed))
    #                         text.insert(END, "file md5 Done:{}".format(file_hashed))
    #                         text.see(END)


    #                         for line in virus_def:
    #                             if file_hashed== line.strip():
    #                                 # لو فيه تطابق
    #                                 print("Malware Detected --> file name: {}".format(f))
    #                                 text.insert(END, "Malware Detected --> file name: {}".format(f))
    #                                 text.see(END)
    #                                 infected_list.append(f)


    #                             else:
    #                                 pass


    #                     except Exception as e:
    #                         print(" could not read the file Error: {}".format(e))
    #                         text.insert(END, " could not read the file Error: {}".format(e))
    #                         text.see(END)
    #             except:


    #                 pass

    #         print("Infected files found : {}".format(infected_list))
    #         text.insert(END, "Infected files found : {}".format(infected_list))
    #         text.see(END)
            
    #         # deleteOrnot=str(input("would you like to delete the infected files y=yes n=no (y/n)"))
    #         deleteOrnot = messagebox.askquestion('Delete The Infected Files', 'Would you like to delete the infected files?',
    #                                     icon='warning')
    #         insert_text(str(deleteOrnot))
    #         if deleteOrnot == 'yes':
    #             for infected in infected_list:
    #                 os.remove(infected)
    #                 print("file removed : {}".format(infected))
    #                 text.insert(END, "file removed : {}".format(infected))
    #                 text.see(END)
    #                 insert_text(" See You ...")
    #                 os.system("PAUSE")
    #         else:


    #             print(" See You ...")
    #             insert_text(" See You ...")
    #             os.system("PAUSE")
    #     scan()

    

    # --------------------End KDD Function------------------#

    #--------------------Main Frame ---------------------#

    winFrame = Frame(window, width="1200",height="750",bg="white")
    winFrame.pack()
    winFrame.pack_propagate(0)

    txtFrame = Frame(winFrame, width="1200",height="450",bg="white")
    txtFrame.pack()
    txtFrame.pack_propagate(0)

    # text=Text(winFrame, width=80, height=15)
    # text.insert(END, "")
    # text.pack()

    # # create a Scrollbar and associate it with txt
    # scrollb = Scrollbar(winFrame, command=text.yview)
    # scrollb.grid(row=0, column=1, sticky='nsew')
    # text['yscrollcommand'] = scrollb.set

    # Add a Scrollbar(horizontal)
    v = Scrollbar(txtFrame, orient='vertical')
    v.pack(side=RIGHT, fill='y')

    # Add a text widget
    text = Text(txtFrame, width=80, height=10,font=("Georgia, 24"), yscrollcommand=v.set)
    text.insert(END, "")
    text.pack()
    # text=Text(win, font=("Georgia, 24"), yscrollcommand=v.set)

    # Add some text in the text widget
    # for i in range(10):
    #     text.insert(END, "Welcome to Tutorialspoint...\n\n")

    # Attach the scrollbar with the text widget
    v.config(command=text.yview)
    text.pack()

    #--------------------Main Frame End ------------------#

    # --------------------KDDCUP99 Button --------------------#

    kddButtonImg = PhotoImage(file="res\\Buttons\\Non-Hoved\\KDDCUP99.png")
    hovKddButtonImg = PhotoImage(file="res\\Buttons\\Hoved\\hovKDDCUP99.png")


    def KddButtonEnterFrame(event):
        kddButton.config(image=hovKddButtonImg)

    def KddButtonLeaveFrame(event):
        kddButton.config(image=kddButtonImg)

    # def threading():
    #     # Call work function
    #     t1=threading.Thread(target=get_KDDCUP99)
    #     t1.start()


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

    scanButtonImg = PhotoImage(file="res\\Buttons\\Non-Hoved\\scan.png")
    hovScanButtonImg = PhotoImage(file="res\\Buttons\\Hoved\\hovScan.png")

    def ScanButtonEnterFrame(event):
        scanButton.config(image=hovScanButtonImg)

    def ScanButtonLeaveFrame(event):
        scanButton.config(image=scanButtonImg)


    scanButton = Label(winFrame, image=scanButtonImg,bg="white",cursor="hand2")
    scanButton.place(x=515, y=500)
    #(x=515, y=500)
    

    scanButton.bind('<Enter>', ScanButtonEnterFrame)
    scanButton.bind('<Leave>', ScanButtonLeaveFrame)
    scanButton.bind("<Button-1>", lambda e:threading.Thread(target=Scan).start())

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