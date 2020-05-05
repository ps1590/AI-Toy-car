import rechelp as rec

#rec.usePyAudio=False
#rec.init(12345)
words = ['back', 'forward', 'left', 'right', 'stop']

for word in words:
	print('Recording', word)
	for i in range(1, 81):
		input()
		print(i)
		f = open('in/' + word + '/' + word + str(i) + '.wav', 'wb')
		f.write(rec.record(2))
		f.close()
