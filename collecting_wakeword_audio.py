"""Script to collect data for wake word training..

    To record environment sound run set seconds to None. This will
    record indefinitely until ctrl + c

    To record for a set amount of time set seconds to whatever you want

    To record interactively (usually for recording your own wake words N times).
    use --interactive mode
"""
import wave
import pyaudio
import os
import argparse


# - Number of channels : number of independent audio channels,
#    example: 1 or 2. meaning audio comming from one or two directions.
# - Sample width : number of bytes of each sample.
# - Frame rate/sample rate : number of samples for each secondes,
#    example: 44,100 Hz.
# - Number of frames/number of sample
# - Values of a frame

#sample width == frame_per_buffer

class Recording_WakeWord:

    def __init__(self, args):
        self.FRAMES_PER_BUFFER = args.frame_per_buffer
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = args.channels
        self.FRAME_RATE = args.frame_rate
        self.secondes = args.secondes
        self.p = pyaudio.PyAudio()

        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.FRAME_RATE,
            input=True,
            frames_per_buffer=self.FRAMES_PER_BUFFER
        )
    
    def Starting(self):
        print("Recording started ...")
        frame = []

        for i in range(0, int(self.FRAME_RATE / self.FRAMES_PER_BUFFER * self.secondes)):
            data = self.stream.read(self.FRAMES_PER_BUFFER)
            frame.append(data)

        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

        return frame
    

    def Saving(self, frame, index, path):
        print(f"Saving file {index}.wave ...")
        save_path = os.path.join(path, "{}.wave".format(index))
        obj = wave.open(save_path, "wb")
        obj.setnchannels(self.CHANNELS)
        obj.setsampwidth(self.p.get_sample_size(self.FORMAT))
        obj.setframerate(self.FRAME_RATE)
        obj.writeframes(b"".join(frame)) #b"" to write in binary
        obj.close()

def Multi_recording(args):
    name = 0
    print()
    try:
        while True:
            Rec = Recording_WakeWord(args)
            frames = Rec.Starting()
            #save_path = os.path.join(args.interactive_save_path, "{}.wave".format(name))
            Rec.Saving(frame=frames, index=name, path=args.interactive_save_path)
            print("Press ctrl + c to exit the multiple recording mode.")
            name += 1
    except KeyboardInterrupt:
        print("Keyboard interuption ")
    except Exception as e:
        print(str(e))

def Single_recording(args):
    Rec = Recording_WakeWord(args)

    frames = Rec.Starting()
    #save_path = os.path.join(args.save_path, "{}.wave".format(args.name))
    Rec.Saving(frame=frames, index=args.name, path=args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
    Script to collect data for wake word training..

    To record environment sound run set seconds to None. This will
    record indefinitely until ctrl + c.

    To record for a set amount of time set seconds to whatever you want.

    To record interactively (usually for recording your own wake words N times)
    use --interactive mode.
    ''')

    parser.add_argument("-s", "--secondes", type=int, default=5,
                        help="how many secondes the recording last.")
    parser.add_argument("-fpb", "--frame_per_buffer", type=int, default=3200,
                        help="number of audio samples that are processed together as a unit within a single buffer.")
    parser.add_argument("-c", "--channels", choices=[1, 2], default=1,
                        help="number of independent audio channels, example: 1 or 2. meaning audio comming from one or two directions.")
    parser.add_argument("-fps", "--frame_rate", type=int, default=16000,
                        help="number of samples for each secondes, example: 44,100 Hz.")
    parser.add_argument("--save_path", type=str, default=None,
                        help="full path to save file. i.e. /to/path/sound.wave")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="sets to interactive mode")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--interactive_save_path", type=str, default=None,
                        help="directory to save all the interactive 2 second samples. i.e. /to/path/")
    group.add_argument("-n", "--name", type=int, default=1,
                       help="name of the file to save")
    
    args = parser.parse_args()

    if args.interactive:
        if args.interactive_save_path is None:
            raise Exception('need to set --interactive_save_path') 
        Multi_recording(args)
    else:
        if args.save_path is None:
            raise Exception('need to set --save_path')
        Single_recording(args)