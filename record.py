from sys import byteorder
from array import array
from struct import pack
import noise_reduce_core as nr
import numpy as np

import time

import scipy.io.wavfile as wav


import pyaudio
import wave

THRESHOLD = 1300
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 8000

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def noise_reduce(snd_data):
    data_numpy = np.array(snd_data)
    reduced_noise = nr.reduce_noise(audio_clip=data_numpy.astype('float32'), noise_clip=data_numpy.astype('float32'))
    # reduced_noise = nr.reduce_noise(audio_clip=reduced_noise.astype('float32'), noise_clip=reduced_noise.astype('float32'))

    #wav.write("test_files/reduced-noise.wav", RATE, reduced_noise.astype("int16"))
    return np.asarray(reduced_noise)


def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i) > THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')
    
    start = time.time()

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True
            start = time.time()

        if snd_started:
            now = time.time()
            print(now - start)
            if now - start > 0.8:
                break

        if snd_started and num_silent > 20:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    # noise reduce
    r = normalize(r)
    r = noise_reduce(r)
    r = trim(r)

    return sample_width, r





def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data= record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


if __name__ == '__main__':
    # await_first_sound()
    print("please speak a word into the microphone")
    record_to_file("test_files/test.wav")
    print("done - result written to test.wav")
