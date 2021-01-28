from threading import Thread
from record import record_to_file
from features import mfcc
from anntester_single import *
import scipy.io.wavfile as wav
import playsound as plsnd
import requests as req

HOST = "http://192.168.137.163:3000"


if __name__ == '__main__':
    testNet = testInit()

    num_loop = 0
    filename="test_files/test.wav"

    while True:
        # Record to file
        num_loop += 1
        print("please speak a word into the microphone", num_loop)
        record_to_file(filename)

        # Feed into ANN
        inputArray = extractFeature(filename)
        res = feedToNetwork(inputArray,testNet)

        if(res == 0):
            # ban can giup gi?
            plsnd.playsound("speak_out_files/bancangiupgi.wav")
            print("Ban can giup gi? ...")
            
            record_to_file(filename)

            inputArray = extractFeature(filename)
            res = feedToNetwork(inputArray,testNet)

            if res == 1:
                req.get(HOST + "/loa?data=1")
                plsnd.playsound("speak_out_files/dabatden.wav")

            elif res == 2:
                req.get(HOST + "/loa?data=2")
                plsnd.playsound("speak_out_files/dabatquat.wav")

            elif res == 3:
                req.get(HOST + "/loa?data=3")
                plsnd.playsound("speak_out_files/datatden.wav")

            elif res == 4:
                req.get(HOST + "/loa?data=4")
                plsnd.playsound("speak_out_files/datatquat.wav")
