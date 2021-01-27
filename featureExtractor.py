from __future__ import division
from features import mfcc
from operator import add
import scipy.io.wavfile as wav
import numpy as np

# words = ['alo', 'batden', 'tatden', 'batquat', 'tatquat']
words = ['alo', 'batdenbancong', 'batdennhabep', 'batdenphongkhach', 'batdenphongngu', 'batdentoilet', 'batlovisong', 'batquatphongkhach', 'batquatphongngu', 'battiviphongkhach', 'battiviphongngu', 'dongcuanhabep', 'dongcuanhavesinh', 'dongcuaphongkhach', 'dongcuaphongngu', 'mocuanhabep', 'mocuanhavesinh', 'mocuaphongkhach', 'mocuaphongngu', 'tatdenbancong', 'tatdennhabep', 'tatdenphongkhach', 'tatdenphongngu', 'tatdentoilet', 'tatlovisong', 'tatquatphongkhach', 'tatquatphongngu', 'tattiviphongkhach']


for x in range(len(words)):
	fileString = words[x]+"_mfcc"
	data = []
	for i in range(10):
		(rate,sig) = wav.read("data_training/"+ words[x] + "-" + str(i+1) + ".wav")
		print ("Reading: " + words[x] + "-" + str(i+1) + ".wav")
		duration = len(sig)/rate
		mfcc_feat = mfcc(sig,rate,winlen=duration/20,winstep=duration/20)
		s = mfcc_feat[:20]
		st = []
		for elem in s:
			st.extend(elem)
		
		st /= np.max(np.abs(st),axis=0)
		data.append(st)
		print(st)
		
	with open("mfccData2/" + fileString+ ".npy", 'wb') as outfile:
   		np.save(outfile,data)




