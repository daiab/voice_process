import sys
import time
import os
import glob
import numpy
import pickle as cPickle
import aifc
import math
from numpy import NaN, Inf, arange, isscalar, array
from scipy.fftpack import rfft
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
from scipy.signal import fftconvolve
from matplotlib.mlab import find
import matplotlib.pyplot as plt
from scipy import linalg as la
import audioTrainTest as aT
import audio_basic_io
import utilities
from scipy.signal import lfilter, hamming

# from scikits.talkbox import lpc

# reload(sys)
# sys.setdefaultencoding('utf8')

eps = 0.00000001

""" Time-domain audio features """


def stZCR(frame):
    """Computes zero crossing rate of frame"""
    count = len(frame)
    countZ = numpy.sum(numpy.abs(numpy.diff(numpy.sign(frame)))) / 2
    return (numpy.float64(countZ) / numpy.float64(count - 1.0))


def stEnergy(frame):
    """Computes signal energy of frame"""
    return numpy.sum(frame ** 2) / numpy.float64(len(frame))


def stEnergyEntropy(frame, numOfShortBlocks=10):
    """Computes entropy of energy"""
    Eol = numpy.sum(frame ** 2)  # total frame energy
    L = len(frame)
    subWinLength = int(numpy.floor(L / numOfShortBlocks))
    if L != subWinLength * numOfShortBlocks:
        frame = frame[0:subWinLength * numOfShortBlocks]
    # subWindows is of size [numOfShortBlocks x L]
    subWindows = frame.reshape(subWinLength, numOfShortBlocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = numpy.sum(subWindows ** 2, axis=0) / (Eol + eps)

    # Compute entropy of the normalized sub-frame energies:
    Entropy = -numpy.sum(s * numpy.log2(s + eps))
    return Entropy


""" Frequency-domain audio features """


def stSpectralCentroidAndSpread(X, fs):
    """Computes spectral centroid of frame (given abs(FFT))"""
    ind = (numpy.arange(1, len(X) + 1)) * (fs / (2.0 * len(X)))

    Xt = X.copy()
    Xt = Xt / Xt.max()
    NUM = numpy.sum(ind * Xt)
    DEN = numpy.sum(Xt) + eps

    # Centroid:
    C = (NUM / DEN)

    # Spread:
    S = numpy.sqrt(numpy.sum(((ind - C) ** 2) * Xt) / DEN)

    # Normalize:
    C = C / (fs / 2.0)
    S = S / (fs / 2.0)

    return (C, S)


def stSpectralEntropy(X, numOfShortBlocks=10):
    """Computes the spectral entropy"""
    L = len(X)  # number of frame samples
    Eol = numpy.sum(X ** 2)  # total spectral energy

    subWinLength = int(numpy.floor(L / numOfShortBlocks))  # length of sub-frame
    if L != subWinLength * numOfShortBlocks:
        X = X[0:subWinLength * numOfShortBlocks]

    subWindows = X.reshape(subWinLength, numOfShortBlocks, order='F').copy()  # define sub-frames (using matrix reshape)
    s = numpy.sum(subWindows ** 2, axis=0) / (Eol + eps)  # compute spectral sub-energies
    En = -numpy.sum(s * numpy.log2(s + eps))  # compute spectral entropy

    return En


def stSpectralFlux(X, Xprev):
    """
    Computes the spectral flux feature of the current frame
    ARGUMENTS:
        X:        the abs(fft) of the current frame
        Xpre:        the abs(fft) of the previous frame
    """
    # compute the spectral flux as the sum of square distances:
    sumX = numpy.sum(X + eps)
    sumPrevX = numpy.sum(Xprev + eps)
    F = numpy.sum((X / sumX - Xprev / sumPrevX) ** 2)

    return F


def stSpectralRollOff(X, c, fs):
    """Computes spectral roll-off"""
    totalEnergy = numpy.sum(X ** 2)
    fftLength = len(X)
    Thres = c * totalEnergy
    # Ffind the spectral rolloff as the frequency position where the respective spectral energy is equal to c*totalEnergy
    CumSum = numpy.cumsum(X ** 2) + eps
    [a, ] = numpy.nonzero(CumSum > Thres)
    if len(a) > 0:
        mC = numpy.float64(a[0]) / (float(fftLength))
    else:
        mC = 0.0
    return (mC)


def stHarmonic(frame, fs):
    """
    Computes harmonic ratio and pitch
    """
    M = numpy.round(0.016 * fs) - 1
    R = numpy.correlate(frame, frame, mode='full')

    g = R[len(frame) - 1]
    R = R[len(frame):-1]

    # estimate m0 (as the first zero crossing of R)
    [a, ] = numpy.nonzero(numpy.diff(numpy.sign(R)))

    if len(a) == 0:
        m0 = len(R) - 1
    else:
        m0 = a[0]
    if M > len(R):
        M = len(R) - 1

    Gamma = numpy.zeros((M), dtype=numpy.float64)
    CSum = numpy.cumsum(frame ** 2)
    Gamma[m0:M] = R[m0:M] / (numpy.sqrt((g * CSum[M:m0:-1])) + eps)

    ZCR = stZCR(Gamma)

    if ZCR > 0.15:
        HR = 0.0
        f0 = 0.0
    else:
        if len(Gamma) == 0:
            HR = 1.0
            blag = 0.0
            Gamma = numpy.zeros((M), dtype=numpy.float64)
        else:
            HR = numpy.max(Gamma)
            blag = numpy.argmax(Gamma)

        # Get fundamental frequency:
        f0 = fs / (blag + eps)
        if f0 > 5000:
            f0 = 0.0
        if HR < 0.1:
            f0 = 0.0

    return (HR, f0)


def mfccInitFilterBanks(fs, nfft):
    """
    Computes the triangular filterbank for MFCC computation (used in the stFeatureExtraction function before the stMFCC function call)
    This function is taken from the scikits.talkbox library (MIT Licence):
    https://pypi.python.org/pypi/scikits.talkbox
    """

    # filter bank params:
    lowfreq = 133.33
    linsc = 200 / 3.
    logsc = 1.0711703
    numLinFiltTotal = 13
    numLogFilt = 27

    if fs < 8000:
        nlogfil = 5

    # Total number of filters
    nFiltTotal = numLinFiltTotal + numLogFilt
    nFiltTotal = int(nFiltTotal)
    nfft = int(nfft)

    # Compute frequency points of the triangle:
    freqs = numpy.zeros(nFiltTotal + 2)
    freqs[:numLinFiltTotal] = lowfreq + numpy.arange(numLinFiltTotal) * linsc
    freqs[numLinFiltTotal:] = freqs[numLinFiltTotal - 1] * logsc ** numpy.arange(1, numLogFilt + 3)
    heights = 2. / (freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    nFiltTotal = int(nFiltTotal)
    fbank = numpy.zeros((nFiltTotal, nfft))
    nfreqs = numpy.arange(nfft) / (1. * nfft) * fs

    for i in range(nFiltTotal):
        lowTrFreq = freqs[i]
        cenTrFreq = freqs[i + 1]
        highTrFreq = freqs[i + 2]

        lid = numpy.arange(numpy.floor(lowTrFreq * nfft / fs) + 1, numpy.floor(cenTrFreq * nfft / fs) + 1,
                           dtype=numpy.int)
        lslope = heights[i] / (cenTrFreq - lowTrFreq)
        rid = numpy.arange(numpy.floor(cenTrFreq * nfft / fs) + 1, numpy.floor(highTrFreq * nfft / fs) + 1,
                           dtype=numpy.int)
        rslope = heights[i] / (highTrFreq - cenTrFreq)
        fbank[i][lid] = lslope * (nfreqs[lid] - lowTrFreq)
        fbank[i][rid] = rslope * (highTrFreq - nfreqs[rid])

    return fbank, freqs


def stMFCC(X, fbank, nceps):
    """
    Computes the MFCCs of a frame, given the fft mag

    ARGUMENTS:
        X:        fft magnitude abs(FFT)
        fbank:    filter bank (see mfccInitFilterBanks)
    RETURN
        ceps:     MFCCs (13 element vector)

    Note:    MFCC calculation is, in general, taken from the scikits.talkbox library (MIT Licence),
    #    with a small number of modifications to make it more compact and suitable for the pyAudioAnalysis Lib
    """

    mspec = numpy.log10(numpy.dot(X, fbank.T) + eps)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:nceps]
    return ceps


def stChromaFeaturesInit(nfft, fs):
    """
    This function initializes the chroma matrices used in the calculation of the chroma features
    """
    nfft = int(nfft)
    fs = int(fs)
    freqs = numpy.array([((f + 1) * fs) / (2 * nfft) for f in range(nfft)])
    Cp = 27.50
    nChroma = numpy.round(12.0 * numpy.log2(freqs / Cp)).astype(int)

    nFreqsPerChroma = numpy.zeros((nChroma.shape[0],))

    uChroma = numpy.unique(nChroma)
    for u in uChroma:
        idx = numpy.nonzero(nChroma == u)
        nFreqsPerChroma[idx] = idx[0].shape

    return nChroma, nFreqsPerChroma


def stChromaFeatures(X, fs, nChroma, nFreqsPerChroma):
    # TODO: 1 complexity
    # TODO: 2 bug with large windows

    chromaNames = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    spec = X ** 2
    if nChroma.max() < nChroma.shape[0]:
        C = numpy.zeros((nChroma.shape[0],))
        C[nChroma] = spec
        C /= nFreqsPerChroma[nChroma]
    else:
        I = numpy.nonzero(nChroma > nChroma.shape[0])[0][0]
        C = numpy.zeros((nChroma.shape[0],))
        C[nChroma[0:I - 1]] = spec
        C /= nFreqsPerChroma
    finalC = numpy.zeros((12, 1))
    newD = int(numpy.ceil(C.shape[0] / 12.0) * 12)
    C2 = numpy.zeros((newD,))
    C2[0:C.shape[0]] = C
    C2 = C2.reshape(int(C2.shape[0] / 12), 12)
    # for i in range(12):
    #    finalC[i] = numpy.sum(C[i:C.shape[0]:12])
    finalC = numpy.matrix(numpy.sum(C2, axis=0)).T
    finalC /= spec.sum()

    #    ax = plt.gca()
    #    plt.hold(False)
    #    plt.plot(finalC)
    #    ax.set_xticks(range(len(chromaNames)))
    #    ax.set_xticklabels(chromaNames)
    #    xaxis = numpy.arange(0, 0.02, 0.01);
    #    ax.set_yticks(range(len(xaxis)))
    #    ax.set_yticklabels(xaxis)
    #    plt.show(block=False)
    #    plt.draw()

    return chromaNames, finalC


def stChromagram(signal, Fs, Win, Step, PLOT=False):
    """
    Short-term FFT mag for spectogram estimation:
    Returns:
        a numpy array (nFFT x numOfShortTermWindows)
    ARGUMENTS:
        signal:      the input signal samples
        Fs:          the sampling freq (in Hz)
        Win:         the short-term window size (in samples)
        Step:        the short-term window step (in samples)
        PLOT:        flag, 1 if results are to be ploted
    RETURNS:
    """
    Win = int(Win)
    Step = int(Step)
    signal = numpy.double(signal)
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX - DC)

    N = len(signal)  # total number of signals
    curPos = 0
    countFrames = 0
    nfft = int(Win / 2)
    nChroma, nFreqsPerChroma = stChromaFeaturesInit(nfft, Fs)
    chromaGram = numpy.array([], dtype=numpy.float64)

    while (curPos + Win - 1 < N):
        countFrames += 1
        x = signal[curPos:curPos + Win]
        curPos = curPos + Step
        X = abs(fft(x))
        X = X[0:nfft]
        X = X / len(X)
        chromaNames, C = stChromaFeatures(X, Fs, nChroma, nFreqsPerChroma)
        C = C[:, 0]
        if countFrames == 1:
            chromaGram = C.T
        else:
            chromaGram = numpy.vstack((chromaGram, C.T))
    FreqAxis = chromaNames
    TimeAxis = [(t * Step) / Fs for t in range(chromaGram.shape[0])]

    if (PLOT):
        fig, ax = plt.subplots()
        chromaGramToPlot = chromaGram.transpose()[::-1, :]
        Ratio = chromaGramToPlot.shape[1] / (3 * chromaGramToPlot.shape[0])
        if Ratio < 1:
            Ratio = 1
        chromaGramToPlot = numpy.repeat(chromaGramToPlot, Ratio, axis=0)
        imgplot = plt.imshow(chromaGramToPlot)
        Fstep = int(nfft / 5.0)
        #        FreqTicks = range(0, int(nfft) + Fstep, Fstep)
        #        FreqTicksLabels = [str(Fs/2-int((f*Fs) / (2*nfft))) for f in FreqTicks]
        ax.set_yticks(range(Ratio / 2, len(FreqAxis) * Ratio, Ratio))
        ax.set_yticklabels(FreqAxis[::-1])
        TStep = countFrames / 3
        TimeTicks = range(0, countFrames, TStep)
        TimeTicksLabels = ['%.2f' % (float(t * Step) / Fs) for t in TimeTicks]
        ax.set_xticks(TimeTicks)
        ax.set_xticklabels(TimeTicksLabels)
        ax.set_xlabel('time (secs)')
        imgplot.set_cmap('jet')
        plt.colorbar()
        plt.show()

    return (chromaGram, TimeAxis, FreqAxis)


def phormants(x, Fs):
    N = len(x)
    w = numpy.hamming(N)

    # Apply window and high pass filter.
    x1 = x * w
    x1 = lfilter([1], [1., 0.63], x1)

    # Get LPC.    
    ncoeff = 2 + Fs / 1000
    A, e, k = lpc(x1, ncoeff)
    # A, e, k = lpc(x1, 8)

    # Get roots.
    rts = numpy.roots(A)
    rts = [r for r in rts if numpy.imag(r) >= 0]

    # Get angles.
    angz = numpy.arctan2(numpy.imag(rts), numpy.real(rts))

    # Get frequencies.    
    frqs = sorted(angz * (Fs / (2 * math.pi)))

    return frqs


def beatExtraction(stFeatures, winSize, PLOT=False):
    """
    This function extracts an estimate of the beat rate for a musical signal.
    ARGUMENTS:
     - stFeatures:     a numpy array (numOfFeatures x numOfShortTermWindows)
     - winSize:        window size in seconds
    RETURNS:
     - BPM:            estimates of beats per minute
     - Ratio:          a confidence measure
    """

    # Features that are related to the beat tracking task:
    toWatch = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    maxBeatTime = int(round(2.0 / winSize))
    HistAll = numpy.zeros((maxBeatTime,))
    for ii, i in enumerate(toWatch):  # for each feature
        DifThres = 2.0 * (
        numpy.abs(stFeatures[i, 0:-1] - stFeatures[i, 1::])).mean()  # dif threshold (3 x Mean of Difs)
        if DifThres <= 0:
            DifThres = 0.0000000000000001
        [pos1, _] = utilities.peakdet(stFeatures[i, :], DifThres)  # detect local maxima
        posDifs = []  # compute histograms of local maxima changes
        for j in range(len(pos1) - 1):
            posDifs.append(pos1[j + 1] - pos1[j])
        [HistTimes, HistEdges] = numpy.histogram(posDifs, numpy.arange(0.5, maxBeatTime + 1.5))
        HistCenters = (HistEdges[0:-1] + HistEdges[1::]) / 2.0
        HistTimes = HistTimes.astype(float) / stFeatures.shape[1]
        HistAll += HistTimes
        if PLOT:
            plt.subplot(9, 2, ii + 1)
            plt.plot(stFeatures[i, :], 'k')
            for k in pos1:
                plt.plot(k, stFeatures[i, k], 'k*')
            f1 = plt.gca()
            f1.axes.get_xaxis().set_ticks([])
            f1.axes.get_yaxis().set_ticks([])

    if PLOT:
        plt.show(block=False)
        plt.figure()

    # Get beat as the argmax of the agregated histogram:
    I = numpy.argmax(HistAll)
    BPMs = 60 / (HistCenters * winSize)
    BPM = BPMs[I]
    # ... and the beat ratio:
    Ratio = HistAll[I] / HistAll.sum()

    if PLOT:
        # filter out >500 beats from plotting:
        HistAll = HistAll[BPMs < 500]
        BPMs = BPMs[BPMs < 500]

        plt.plot(BPMs, HistAll, 'k')
        plt.xlabel('Beats per minute')
        plt.ylabel('Freq Count')
        plt.show(block=True)

    return BPM, Ratio


def stSpectogram(signal, Fs, Win, Step, PLOT=False):
    """
    Short-term FFT mag for spectogram estimation:
    Returns:
        a numpy array (nFFT x numOfShortTermWindows)
    ARGUMENTS:
        signal:      the input signal samples
        Fs:          the sampling freq (in Hz)
        Win:         the short-term window size (in samples)
        Step:        the short-term window step (in samples)
        PLOT:        flag, 1 if results are to be ploted
    RETURNS:
    """
    Win = int(Win)
    Step = int(Step)
    signal = numpy.double(signal)
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX - DC)

    N = len(signal)  # total number of signals
    curPos = 0
    countFrames = 0
    nfft = int(Win / 2)
    specgram = numpy.array([], dtype=numpy.float64)

    while curPos + Win - 1 < N:
        countFrames += 1
        x = signal[curPos:curPos + Win]
        curPos = curPos + Step
        X = abs(fft(x))
        X = X[0:nfft]
        X = X / len(X)

        if countFrames == 1:
            specgram = X ** 2
        else:
            specgram = numpy.vstack((specgram, X))

    FreqAxis = [((f + 1) * Fs) / (2 * nfft) for f in range(specgram.shape[1])]
    TimeAxis = [(t * Step) / Fs for t in range(specgram.shape[0])]

    if (PLOT):
        fig, ax = plt.subplots()
        imgplot = plt.imshow(specgram.transpose()[::-1, :])
        Fstep = int(nfft / 5.0)
        FreqTicks = range(0, int(nfft) + Fstep, Fstep)
        FreqTicksLabels = [str(Fs / 2 - int((f * Fs) / (2 * nfft))) for f in FreqTicks]
        ax.set_yticks(FreqTicks)
        ax.set_yticklabels(FreqTicksLabels)
        TStep = countFrames / 3
        TStep = int(TStep)
        TimeTicks = range(0, countFrames, TStep)
        TimeTicksLabels = ['%.2f' % (float(t * Step) / Fs) for t in TimeTicks]
        ax.set_xticks(TimeTicks)
        ax.set_xticklabels(TimeTicksLabels)
        ax.set_xlabel('time (secs)')
        ax.set_ylabel('freq (Hz)')
        imgplot.set_cmap('jet')
        plt.colorbar()
        plt.show()

    return specgram, TimeAxis, FreqAxis


""" Windowing and feature extraction """


def st_feature_extraction(signal, fr, win, step):
    """
    This function implements the short-term windowing process.
    For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a numpy matrix.

    ARGUMENTS
        signal:       the input signal samples
        Fs:           the sampling freq (in Hz)
        Win:          the short-term window size (in samples)
        Step:         the short-term window step (in samples)
    RETURNS
        stFeatures:   a numpy array (numOfFeatures x numOfShortTermWindows)
    """

    win = int(win)
    step = int(step)
    signal = numpy.double(signal)

    # Signal normalization
    signal = signal / (2.0 ** 15)
    dc = signal.mean()
    max_value = (numpy.abs(signal)).max()
    signal = (signal - dc) / (max_value + 0.0000000001)

    print(signal.shape)
    n = len(signal)  # total number of samples
    cur_pos = 0
    count_frames = 0
    n_FFT = win / 2

    fbank, freqs = mfccInitFilterBanks(fr, n_FFT)  # compute the triangular filter banks used in the mfcc calculation
    n_chroma, n_freqs_per_chroma = stChromaFeaturesInit(n_FFT, fr)

    num_of_time_spectral_features = 8
    num_of_harmonic_features = 0
    nceps = 13
    num_of_chroma_features = 13
    total_num_of_features = num_of_time_spectral_features + nceps + num_of_harmonic_features + num_of_chroma_features
    # totalNumOfFeatures = numOfTimeSpectralFeatures + nceps + numOfHarmonicFeatures

    st_features = []
    while cur_pos + win - 1 < n:  # for each short-term window until the end of signal
        count_frames += 1
        x = signal[cur_pos:cur_pos + win]  # get current window
        cur_pos = cur_pos + step  # update window position
        X = abs(fft(x))  # get fft magnitude
        n_FFT = int(n_FFT)
        X = X[0:n_FFT]  # normalize fft
        X = X / len(X)
        if count_frames == 1:
            Xprev = X.copy()  # keep previous fft mag (used in spectral flux)
        cur_FV = numpy.zeros((total_num_of_features, 1))
        cur_FV[0] = stZCR(x)  # zero crossing rate
        cur_FV[1] = stEnergy(x)  # short-term energy
        cur_FV[2] = stEnergyEntropy(x)  # short-term entropy of energy
        cur_FV[3], cur_FV[4] = stSpectralCentroidAndSpread(X, fr)  # spectral centroid and spread
        cur_FV[5] = stSpectralEntropy(X)  # spectral entropy
        cur_FV[6] = stSpectralFlux(X, Xprev)  # spectral flux
        cur_FV[7] = stSpectralRollOff(X, 0.90, fr)  # spectral rolloff
        cur_FV[num_of_time_spectral_features:num_of_time_spectral_features + nceps, 0] = \
            stMFCC(X, fbank, nceps).copy()  # MFCCs

        chroma_names, chroma_f = stChromaFeatures(X, fr, n_chroma, n_freqs_per_chroma)
        cur_FV[num_of_time_spectral_features + nceps: num_of_time_spectral_features + nceps + num_of_chroma_features - 1] = chroma_f
        cur_FV[num_of_time_spectral_features + nceps + num_of_chroma_features - 1] = chroma_f.std()
        st_features.append(cur_FV)
        # delta features
        '''
        if countFrames>1:
            delta = curFV - prevFV
            curFVFinal = numpy.concatenate((curFV, delta))            
        else:
            curFVFinal = numpy.concatenate((curFV, curFV))
        prevFV = curFV
        stFeatures.append(curFVFinal)        
        '''
        # end of delta
        Xprev = X.copy()

    st_features = numpy.concatenate(st_features, 1)
    return st_features


def mt_feature_extraction(signal, fr, mt_win, mt_step, st_win, st_step):
    """
    Mid-term feature extraction
    """
    mt_win_ratio = int(round(mt_win / st_step))
    mt_step_ratio = int(round(mt_step / st_step))

    st_features = st_feature_extraction(signal, fr, st_win, st_step)
    num_of_features = len(st_features)  # =34
    num_of_statistics = 2

    mt_features = []
    for i in range(num_of_statistics * num_of_features):
        mt_features.append([])

    for i in range(num_of_features):  # for each of the short-term features:
        cur_pos = 0
        n = len(st_features[i])
        while cur_pos < n:
            n1 = cur_pos
            n2 = cur_pos + mt_win_ratio
            if n2 > n:
                n2 = n
            cur_st_features = st_features[i][n1:n2]

            mt_features[i].append(numpy.mean(cur_st_features))
            mt_features[i + num_of_features].append(numpy.std(cur_st_features))
            cur_pos += mt_step_ratio
    return numpy.array(mt_features), st_features


# TODO
def stFeatureSpeed(signal, Fs, Win, Step):
    signal = numpy.double(signal)
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / MAX
    # print (numpy.abs(signal)).max()

    N = len(signal)  # total number of signals
    curPos = 0
    countFrames = 0

    lowfreq = 133.33
    linsc = 200 / 3.
    logsc = 1.0711703
    nlinfil = 13
    nlogfil = 27
    nceps = 13
    nfil = nlinfil + nlogfil
    nfft = Win / 2
    if Fs < 8000:
        nlogfil = 5
        nfil = nlinfil + nlogfil
        nfft = Win / 2

    # compute filter banks for mfcc:
    [fbank, freqs] = mfccInitFilterBanks(Fs, nfft, lowfreq, linsc, logsc, nlinfil, nlogfil)

    numOfTimeSpectralFeatures = 8
    numOfHarmonicFeatures = 1
    totalNumOfFeatures = numOfTimeSpectralFeatures + nceps + numOfHarmonicFeatures
    # stFeatures = numpy.array([], dtype=numpy.float64)
    stFeatures = []

    while (curPos + Win - 1 < N):
        countFrames += 1
        x = signal[curPos:curPos + Win]
        curPos = curPos + Step
        X = abs(fft(x))
        X = X[0:nfft]
        X = X / len(X)
        Ex = 0.0
        El = 0.0
        X[0:4] = 0
        #        M = numpy.round(0.016 * fs) - 1
        #        R = numpy.correlate(frame, frame, mode='full')
        stFeatures.append(stHarmonic(x, Fs))
    #        for i in range(len(X)):
    # if (i < (len(X) / 8)) and (i > (len(X)/40)):
    #    Ex += X[i]*X[i]
    # El += X[i]*X[i]
    #        stFeatures.append(Ex / El)
    #        stFeatures.append(numpy.argmax(X))
    #        if curFV[numOfTimeSpectralFeatures+nceps+1]>0:
    #            print curFV[numOfTimeSpectralFeatures+nceps], curFV[numOfTimeSpectralFeatures+nceps+1]
    return numpy.array(stFeatures)


""" Feature Extraction Wrappers

 - The first two feature extraction wrappers are used to extract long-term averaged
   audio features for a list of WAV files stored in a given category.
   It is important to note that, one single feature is extracted per WAV file (not the whole sequence of feature vectors)

 """


def dirWavFeatureExtraction(dirName, mtWin, mtStep, stWin, stStep, computeBEAT=False):
    """
    This function extracts the mid-term features of the WAVE files of a particular folder.

    The resulting feature vector is extracted by long-term averaging the mid-term features.
    Therefore ONE FEATURE VECTOR is extracted for each WAV file.

    ARGUMENTS:
        - dirName:        the path of the WAVE directory
        - mtWin, mtStep:    mid-term window and step (in seconds)
        - stWin, stStep:    short-term window and step (in seconds)
    """

    allMtFeatures = numpy.array([])
    processingTimes = []

    types = ('*.wav', '*.aif', '*.aiff', '*.mp3', '*.au')
    wavFilesList = []
    for files in types:
        wavFilesList.extend(glob.glob(os.path.join(dirName, files)))

    wavFilesList = sorted(wavFilesList)
    wavFilesList2 = []
    for i, wavFile in enumerate(wavFilesList):
        print("Analyzing file {0:d} of {1:d}: {2:s}".format(i + 1, len(wavFilesList), wavFile.encode('utf-8')))
        if os.stat(wavFile).st_size == 0:
            print("   (EMPTY FILE -- SKIPPING)")
            continue
        [Fs, x] = audio_basic_io.read_audio_file(wavFile)  # read file
        if isinstance(x, int):
            continue

        t1 = time.clock()
        x = audio_basic_io.stereo2mono(x)  # convert stereo to mono
        if x.shape[0] < float(Fs) / 10:
            print("  (AUDIO FILE TOO SMALL - SKIPPING)")
            continue
        wavFilesList2.append(wavFile)
        if computeBEAT:  # mid-term feature extraction for current file
            [MidTermFeatures, stFeatures] = mt_feature_extraction(x, Fs, round(mtWin * Fs), round(mtStep * Fs),
                                                                  round(Fs * stWin), round(Fs * stStep))
            [beat, beatConf] = beatExtraction(stFeatures, stStep)
        else:
            [MidTermFeatures, _] = mt_feature_extraction(x, Fs, round(mtWin * Fs), round(mtStep * Fs),
                                                         round(Fs * stWin), round(Fs * stStep))

        MidTermFeatures = numpy.transpose(MidTermFeatures)
        MidTermFeatures = MidTermFeatures.mean(axis=0)  # long term averaging of mid-term statistics
        if (not numpy.isnan(MidTermFeatures).any()) and (not numpy.isinf(MidTermFeatures).any()):
            if computeBEAT:
                MidTermFeatures = numpy.append(MidTermFeatures, beat)
                MidTermFeatures = numpy.append(MidTermFeatures, beatConf)
            if len(allMtFeatures) == 0:  # append feature vector
                allMtFeatures = MidTermFeatures
            else:
                allMtFeatures = numpy.vstack((allMtFeatures, MidTermFeatures))
            t2 = time.clock()
            duration = float(len(x)) / Fs
            processingTimes.append((t2 - t1) / duration)
    if len(processingTimes) > 0:
        print("Feature extraction complexity ratio: {0:.1f} x realtime".format(
            (1.0 / numpy.mean(numpy.array(processingTimes)))))
    return (allMtFeatures, wavFilesList2)


def dirsWavFeatureExtraction(dirNames, mtWin, mtStep, stWin, stStep, computeBEAT=False):
    '''
    Same as dirWavFeatureExtraction, but instead of a single dir it takes a list of paths as input and returns a list of feature matrices.
    EXAMPLE:
    [features, classNames] =
           a.dirsWavFeatureExtraction(['audioData/classSegmentsRec/noise','audioData/classSegmentsRec/speech',
                                       'audioData/classSegmentsRec/brush-teeth','audioData/classSegmentsRec/shower'], 1, 1, 0.02, 0.02);

    It can be used during the training process of a classification model ,
    in order to get feature matrices from various audio classes (each stored in a seperate path)
    '''

    # feature extraction for each class:
    features = []
    classNames = []
    fileNames = []
    for i, d in enumerate(dirNames):
        [f, fn] = dirWavFeatureExtraction(d, mtWin, mtStep, stWin, stStep, computeBEAT=computeBEAT)
        if f.shape[0] > 0:  # if at least one audio file has been found in the provided folder:
            features.append(f)
            fileNames.append(fn)
            if d[-1] == "/":
                classNames.append(d.split(os.sep)[-2])
            else:
                classNames.append(d.split(os.sep)[-1])
    return features, classNames, fileNames


def dirWavFeatureExtractionNoAveraging(dirName, mtWin, mtStep, stWin, stStep):
    """
    This function extracts the mid-term features of the WAVE files of a particular folder without averaging each file.

    ARGUMENTS:
        - dirName:          the path of the WAVE directory
        - mtWin, mtStep:    mid-term window and step (in seconds)
        - stWin, stStep:    short-term window and step (in seconds)
    RETURNS:
        - X:                A feature matrix
        - Y:                A matrix of file labels
        - filenames:
    """

    allMtFeatures = numpy.array([])
    signalIndices = numpy.array([])
    processingTimes = []

    types = ('*.wav', '*.aif', '*.aiff')
    wavFilesList = []
    for files in types:
        wavFilesList.extend(glob.glob(os.path.join(dirName, files)))

    wavFilesList = sorted(wavFilesList)

    for i, wavFile in enumerate(wavFilesList):
        [Fs, x] = audio_basic_io.read_audio_file(wavFile)  # read file
        if isinstance(x, int):
            continue

        x = audio_basic_io.stereo2mono(x)  # convert stereo to mono
        [MidTermFeatures, _] = mt_feature_extraction(x, Fs, round(mtWin * Fs), round(mtStep * Fs), round(Fs * stWin),
                                                     round(Fs * stStep))  # mid-term feature

        MidTermFeatures = numpy.transpose(MidTermFeatures)
        #        MidTermFeatures = MidTermFeatures.mean(axis=0)        # long term averaging of mid-term statistics
        if len(allMtFeatures) == 0:  # append feature vector
            allMtFeatures = MidTermFeatures
            signalIndices = numpy.zeros((MidTermFeatures.shape[0],))
        else:
            allMtFeatures = numpy.vstack((allMtFeatures, MidTermFeatures))
            signalIndices = numpy.append(signalIndices, i * numpy.ones((MidTermFeatures.shape[0],)))

    return (allMtFeatures, signalIndices, wavFilesList)


# The following two feature extraction wrappers extract features for given audio files, however
# NO LONG-TERM AVERAGING is performed. Therefore, the output for each audio file is NOT A SINGLE FEATURE VECTOR
# but a whole feature matrix.
#
# Also, another difference between the following two wrappers and the previous is that they NO LONG-TERM AVERAGING IS PERFORMED.
# In other words, the WAV files in these functions are not used as uniform samples that need to be averaged but as sequences

def mtFeatureExtractionToFile(fileName, midTermSize, midTermStep, shortTermSize, shortTermStep, outPutFile,
                              storeStFeatures=False, storeToCSV=False, PLOT=False):
    """
    This function is used as a wrapper to:
    a) read the content of a WAV file
    b) perform mid-term feature extraction on that signal
    c) write the mid-term feature sequences to a numpy file
    """
    [Fs, x] = audio_basic_io.read_audio_file(fileName)  # read the wav file
    x = audio_basic_io.stereo2mono(x)  # convert to MONO if required
    if storeStFeatures:
        [mtF, stF] = mt_feature_extraction(x, Fs, round(Fs * midTermSize), round(Fs * midTermStep),
                                           round(Fs * shortTermSize), round(Fs * shortTermStep))
    else:
        [mtF, _] = mt_feature_extraction(x, Fs, round(Fs * midTermSize), round(Fs * midTermStep),
                                         round(Fs * shortTermSize), round(Fs * shortTermStep))

    numpy.save(outPutFile, mtF)  # save mt features to numpy file
    if PLOT:
        print("Mid-term numpy file: " + outPutFile + ".npy saved")
    if storeToCSV:
        numpy.savetxt(outPutFile + ".csv", mtF.T, delimiter=",")
        if PLOT:
            print("Mid-term CSV file: " + outPutFile + ".csv saved")

    if storeStFeatures:
        numpy.save(outPutFile + "_st", stF)  # save st features to numpy file
        if PLOT:
            print("Short-term numpy file: " + outPutFile + "_st.npy saved")
        if storeToCSV:
            numpy.savetxt(outPutFile + "_st.csv", stF.T, delimiter=",")  # store st features to CSV file
            if PLOT:
                print("Short-term CSV file: " + outPutFile + "_st.csv saved")


def mtFeatureExtractionToFileDir(dirName, midTermSize, midTermStep, shortTermSize, shortTermStep, storeStFeatures=False,
                                 storeToCSV=False, PLOT=False):
    types = (dirName + os.sep + '*.wav',)
    filesToProcess = []
    for files in types:
        filesToProcess.extend(glob.glob(files))
    for f in filesToProcess:
        outPath = f
        mtFeatureExtractionToFile(f, midTermSize, midTermStep, shortTermSize, shortTermStep, outPath, storeStFeatures,
                                  storeToCSV, PLOT)
