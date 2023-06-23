from tkinter import *
from threading import Thread,Event
from subprocess import call


class KDDController(object):
    def __init__(self):
        self.thread1 = None
        self.thread2 = None
        self.stop_threads = Event()

    def loop1(self):
        while not self.stop_threads.is_set():
            call (["raspivid -n -op 150 -w 640 -h 480 -b 2666666.67 -t 5000 -o test.mp4"],shell=True)
            call (["raspivid -n -op 150 -w 640 -h 480 -b 2666666.67 -t 5000 -o test1.mp4"],shell=True)

    def loop2(self):
        while not self.stop_threads.is_set():
            call (["arecord -D plughw:1 --duration=5 -f cd -vv rectest.wav"],shell=True)
            call (["arecord -D plughw:1 --duration=5 -f cd -vv rectest1.wav"],shell=True)

    def combine(self):
        self.stop_threads.clear()
        self.thread1 = Thread(target = self.loop1)
        self.thread2 = Thread(target = self.loop2)
        self.thread1.start()
        self.thread2.start()

    def stop(self):
        self.stop_threads.set()
        self.thread1.join()
        self.thread2.join()
        self.thread1 = None
        self.thread2 = None

control = KDDController()
btn1 = Button(tk, text="Start Recording", width=16, height=5, command=control.combine)
btn1.grid(row=2,column=0)
btn2 = Button(tk, text="Stop Recording", width=16, height=5, command=control.stop)
btn2.grid(row=3,column=0)