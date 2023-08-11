import wave
import pyaudio


# - Number of channels : number of independent audio channels,
#    example: 1 or 2. meaning audio comming from one or two directions.
# - Sample width : number of bytes of each sample.
# - Frame rate/sample rate : number of samples for each secondes,
#    example: 44,100 Hz.
# - Number of frames/number of sample
# - Values of a frame

#sample width == frame_per_buffer
FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
FRAME_RATE = 16000

p = pyaudio.PyAudio()

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=FRAME_RATE,
    input=True,
    frames_per_buffer=FRAMES_PER_BUFFER
)

print("Start recording")

seconde = 5
frame = []

for i in range(0, int(FRAME_RATE / FRAMES_PER_BUFFER * seconde)):
    data = stream.read(FRAMES_PER_BUFFER)
    frame.append(data)

stream.stop_stream()
stream.close()
p.terminate()

obj = wave.open("output.wave", "wb")
obj.setnchannels(CHANNELS)
obj.setsampwidth(p.get_sample_size(FORMAT))
obj.setframerate(FRAME_RATE)
obj.writeframes(b"".join(frame)) #b"" to write in binary
obj.close()
