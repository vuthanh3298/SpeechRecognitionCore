import noise_reduce_core as nr
import scipy.io.wavfile as wav




words = ['alo', 'batdenbancong', 'batdennhabep', 'batdenphongkhach', 'batdenphongngu', 'batdentoilet', 'batlovisong', 'batquatphongkhach', 'batquatphongngu', 'battiviphongkhach', 'battiviphongngu', 'dongcuanhabep', 'dongcuanhavesinh', 'dongcuaphongkhach', 'dongcuaphongngu', 'mocuanhabep', 'mocuanhavesinh', 'mocuaphongkhach', 'mocuaphongngu', 'tatdenbancong', 'tatdennhabep', 'tatdenphongkhach', 'tatdenphongngu', 'tatdentoilet', 'tatlovisong', 'tatquatphongkhach', 'tatquatphongngu', 'tattiviphongkhach']


for x in range(len(words)):
	fileString = words[x]+"_mfcc"
	data = []
	for i in range(10):
		rate,data = wav.read("data_training/"+ words[x] + "-" + str(i+1) + ".wav")
		if data.ndim > 1:
			data = data[:, 0]

		print ("Transfer to Noise reduce: " + words[x] + "-" + str(i+1) + ".wav")

		reduced_noise = nr.reduce_noise(audio_clip=data.astype('float32'),noise_clip=data.astype('float32'))

		wav.write("training_sets2/" +words[x] + "-" + str(i+1) + ".wav", rate, reduced_noise.astype("int16"))
        
		# with open("training_sets/" +words[x] + "-" + str(i+1) + ".wav", 'wb') as outfile:
		# np.save(outfile, reduced_noise.astype("int16"))
