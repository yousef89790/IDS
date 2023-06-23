from tkinter import *
from tkinter.filedialog import askdirectory
from tkinter import messagebox

import os

import csv

# open Excel ( Database )
import hashlib

# Encrept Password 
from urllib.request import urlopen

# open Url of Previous Password 

import threading



def R2L():

    window = Tk()
    window.title("R2L Attack")

    window.geometry("1200x750")
    window.minsize("1200", "750")
    window.maxsize("1200", "750")

    
    winFrame = Frame(window, width="1200",height="750",bg="white")
    winFrame.pack()
    winFrame.pack_propagate(0)

    txtFrame = Frame(winFrame, width="1200",height="450",bg="white")
    txtFrame.pack()
    txtFrame.pack_propagate(0)

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


    btn1 = Button(winFrame, text="Start", width=16, height=3, command=lambda :threading.Thread(target=get_R2L).start())
    btn1.place(x=470,y=420)

    # txtIP = Entry(winFrame, width=16, height=5,bg="white",cursor="hand2")
    # txtIP.insert(0, 'Enter Target IP')
    # txtIP.place(x=155, y=400)


    #heading label
    # lblIP = Label(winFrame, text= "Target IP :" , font='Verdana 10 bold')
    # lblIP.place(x=80,y=420)

    # lblPort = Label(winFrame, text= "Port :" , font='Verdana 10 bold')
    # lblPort.place(x=80,y=460)

    # # Entry Box
    # target_ip = StringVar()
    # port_value = IntVar()
        
    # ipentry = Entry(winFrame, width=40 , textvariable = target_ip)
    # ipentry.focus()
    # ipentry.place(x=200 , y=423)

    # portentry = Entry(winFrame, width=40 ,textvariable = port_value)
    # portentry.place(x=200 , y=460)


    def insert_text(txt):
        text.insert(END, txt+"\n")
        text.see(END)


    
    def hash(password):

        result = hashlib.sha256(password.encode())

    # convert Password into Hashed

        return result.hexdigest()

        # conver to Hexa digit


    def get_wordlist(url):
        try:
            with urlopen(url) as f:
                wordlist = f.read().decode('utf-8').splitlines()

                # file Reader to download the file And split it into lines 

                return wordlist
        except Exception as e:

            # if  failed 
            
            print(f'failed to get wordlist: {e}')
            insert_text(f'failed to get wordlist: {e}')
            exit(1)


    def get_users(path):
        try:

            result = []
            # cwd = os.getcwd()  # Get the current working directory (cwd)
            # files = os.listdir(cwd)  # Get all the files in that directory
            # print("Files in %r: %s" % (cwd, files))

            with open(path) as f:

                reader = csv.DictReader(f, delimiter=',')

                # Read the Path
                for row in reader:

                    result.append(dict(row))

                    #each Row With
                return result
        except Exception as e:
            print(f'failed to get users: {e}')
            insert_text(f'failed to get users: {e}')
            exit(1)


    def get_hashed_table(path):
        try:
            result = []
            with open(path) as f:
                reader = csv.DictReader(f, delimiter=',')
                for row in reader:
                    result.append(dict(row))
                return result
        except Exception as e:
            print(f'failed to get hashed table: {e}')
            insert_text(f'failed to get hashed table: {e}')
            exit(1)


    def match_hash(users, hashed_table):
        for user in users:
            password_hash = hash(user['password'])
            for row in hashed_table:
                if password_hash == row['hash']:
                    print(f'username: {user["username"]}, password {row["password"]}')
                    insert_text(f'username: {user["username"]}, password {row["password"]}')


    def get_R2L():
        insert_text("R2L Start ............")
        # Reading feature list  D:\\Projects\\Antivirus\\Antivirus_KDDCUP99
        # with open("D:\\Projects\\Antivirus\\Antivirus_KDDCUP99\\dataset\\kddcup.names",'r') as f:
        #     print(f.read())
        #     # insert_text(str(f.read()))
        #     text.insert("1.0", f.read())

        WORDLIST_URL = 'https://raw.githubusercontent.com/berzerk0/Probable-Wordlists/2df55facf06c7742f2038a8f6607ea9071596128/Real-Passwords/Top12Thousand-probable-v2.txt'
        database_file_path = os.path.join(os.path.dirname(__file__), "database.csv")
        DATABASE_PATH = database_file_path
        hashed_table_file_path = os.path.join(os.path.dirname(__file__), "hashed_table.csv")
        HASHED_TABLE_PATH = hashed_table_file_path

        users = get_users(DATABASE_PATH)
        hashed_table = get_hashed_table(HASHED_TABLE_PATH)
        match_hash(users, hashed_table)


    window.mainloop()
