import audioSegmentation as aS
import audio_basic_io
import scipy.io.wavfile as wavfile
import numpy as np

example_file = "/home/daiab/machine_disk/data/voice_identity/dianxin/1.wav"


def seg():
    # [flagsInd, classesAll, acc, CM] = aS.mtFileClassification("data/scottish.wav", "data/svmSM", "svm", True, 'data/scottish.segments')
    flagsInd, classesAll, acc, CM = aS.mtFileClassification(
        example_file,
        "data/svmSM", "svm", True, 'data/scottish.segments')


def extract_feat():
    import audio_basic_io
    import audioFeatureExtraction
    import matplotlib.pyplot as plt
    Fs, x = audio_basic_io.read_audio_file(example_file)
    F = audioFeatureExtraction.st_feature_extraction(x, Fs, 0.050 * Fs, 0.025 * Fs);
    plt.subplot(2, 1, 1)
    plt.plot(F[0, :])
    plt.xlabel('Frame no')
    plt.ylabel('ZCR')
    plt.subplot(2, 1, 2)
    plt.plot(F[1, :])
    plt.xlabel('Frame no')
    plt.ylabel('Energy')
    plt.show()


def plot_spectorgram():
    import audioFeatureExtraction as aF
    Fs, x = audio_basic_io.read_audio_file(example_file)
    x = audio_basic_io.stereo2mono(x)
    specgram, TimeAxis, FreqAxis = aF.stSpectogram(x, Fs,
                                                   round(Fs * 0.040),
                                                   round(Fs * 0.040),
                                                   True)


def train():
    import audioTrainTest as aT
    aT.featureAndTrain(["/home/tyiannak/Desktop/MusicGenre/Classical/", "/home/tyiannak/Desktop/MusicGenre/Electronic/",
                        "/home/tyiannak/Desktop/MusicGenre/Jazz/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep,
                       "svm", "svmMusicGenre3", True)
    aT.featureAndTrain(["/home/tyiannak/Desktop/MusicGenre/Classical/", "/home/tyiannak/Desktop/MusicGenre/Electronic/",
                        "/home/tyiannak/Desktop/MusicGenre/Jazz/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep,
                       "knn", "knnMusicGenre3", True)
    aT.featureAndTrain(["/home/tyiannak/Desktop/MusicGenre/Classical/", "/home/tyiannak/Desktop/MusicGenre/Electronic/",
                        "/home/tyiannak/Desktop/MusicGenre/Jazz/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep,
                       "extratrees", "etMusicGenre3", True)
    aT.featureAndTrain(["/home/tyiannak/Desktop/MusicGenre/Classical/", "/home/tyiannak/Desktop/MusicGenre/Electronic/",
                        "/home/tyiannak/Desktop/MusicGenre/Jazz/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep,
                       "gradientboosting", "gbMusicGenre3", True)
    aT.featureAndTrain(["/home/tyiannak/Desktop/MusicGenre/Classical/", "/home/tyiannak/Desktop/MusicGenre/Electronic/",
                        "/home/tyiannak/Desktop/MusicGenre/Jazz/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep,
                       "randomforest", "rfMusicGenre3", True)
    aT.featureAndTrain(["/home/tyiannak/Desktop/5Class/Silence/", "/home/tyiannak/Desktop/5Class/SpeechMale/",
                        "/home/tyiannak/Desktop/5Class/SpeechFemale/", "/home/tyiannak/Desktop/5Class/ObjectsOther/",
                        "/home/tyiannak/Desktop/5Class/Music/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm",
                       "svm5Classes")
    aT.featureAndTrain(["/home/tyiannak/Desktop/5Class/Silence/", "/home/tyiannak/Desktop/5Class/SpeechMale/",
                        "/home/tyiannak/Desktop/5Class/SpeechFemale/", "/home/tyiannak/Desktop/5Class/ObjectsOther/",
                        "/home/tyiannak/Desktop/5Class/Music/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn",
                       "knn5Classes")
    aT.featureAndTrain(["/home/tyiannak/Desktop/5Class/Silence/", "/home/tyiannak/Desktop/5Class/SpeechMale/",
                        "/home/tyiannak/Desktop/5Class/SpeechFemale/", "/home/tyiannak/Desktop/5Class/ObjectsOther/",
                        "/home/tyiannak/Desktop/5Class/Music/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep,
                       "extratrees", "et5Classes")
    aT.featureAndTrain(["/home/tyiannak/Desktop/5Class/Silence/", "/home/tyiannak/Desktop/5Class/SpeechMale/",
                        "/home/tyiannak/Desktop/5Class/SpeechFemale/", "/home/tyiannak/Desktop/5Class/ObjectsOther/",
                        "/home/tyiannak/Desktop/5Class/Music/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep,
                       "gradientboosting", "gb5Classes")
    aT.featureAndTrain(["/home/tyiannak/Desktop/5Class/Silence/", "/home/tyiannak/Desktop/5Class/SpeechMale/",
                        "/home/tyiannak/Desktop/5Class/SpeechFemale/", "/home/tyiannak/Desktop/5Class/ObjectsOther/",
                        "/home/tyiannak/Desktop/5Class/Music/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep,
                       "randomforest", "rf5Classes")


def speaker_diarization():
    file = '/home/daiab/machine_disk/data/voice_identity/dianxin/1.wav'
    use_LDA = False
    plot = True
    num_speaker = 2
    if use_LDA:
        pos, cls = aS.speaker_diarization(file,
                                          num_speaker,
                                          mt_size=4.0,
                                          mt_step=0.1,
                                          st_win=0.05,
                                          st_step=0.01,
                                          plot=plot)
    else:
        pos, cls = aS.speaker_diarization(file, num_speaker, lda_dim=0, plot=plot)
    fr, x = audio_basic_io.read_audio_file(file)

    sep_voice = [[], []]
    pre_pos = 0
    cut_num = int(x.shape[0] * 0.0001)
    print('cut_num', cut_num)
    for i, c in enumerate(cls):
        c = int(c)
        v_from = pre_pos
        v_to = int(pos[i] * fr)
        sep_voice[c] += x[v_from + cut_num: v_to - cut_num].tolist()
        pre_pos = v_to

    print(len(sep_voice[0]), len(sep_voice[1]))
    wavfile.write('./0.wav', fr, np.array(sep_voice[0], dtype=np.int16))
    wavfile.write('./1.wav', fr, np.array(sep_voice[1], dtype=np.int16))


def remove_silence():
    smoothing = 1.0
    weight = 0.5
    example_file = '/home/yulongwu/d/voice/wav/2nU95KARZwk.wav'
    fr, x = audio_basic_io.read_audio_file(example_file)
    print('x shape', x.shape)
    segment_limits = aS.silenceRemoval(x, fr, 0.05, 0.05,
                                       smoothing, weight, True)
    for i, s in enumerate(segment_limits):
        name = "{0:s}_{1:.3f}-{2:.3f}.wav".format(example_file[0:-4], s[0], s[1])
        wavfile.write(name, fr, x[int(fr * s[0]):int(fr * s[1])])


if __name__ == '__main__':
    remove_silence()
    # speaker_diarization()
    # plot_spectorgram()
    # extract_feat()
