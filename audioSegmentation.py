import numpy as np
import sklearn.cluster
import time
import scipy
import os
import audioFeatureExtraction as aF
import audioTrainTest as aT
import audio_basic_io
import matplotlib.pyplot as plt
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn.discriminant_analysis
import csv
import os.path
import sklearn
import sklearn.cluster
import hmmlearn.hmm
import pickle as cPickle
import glob

""" General utility functions """


def smoothMovingAvg(inputSignal, windowLen=11):
    windowLen = int(windowLen)
    if inputSignal.ndim != 1:
        raise ValueError("")
    if inputSignal.size < windowLen:
        raise ValueError("Input vector needs to be bigger than window size.")
    if windowLen < 3:
        return inputSignal
    s = np.r_[2 * inputSignal[0] - inputSignal[windowLen - 1::-1], inputSignal, 2 * inputSignal[-1] - inputSignal[
                                                                                                      -1:-windowLen:-1]]
    w = np.ones(windowLen, 'd')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[windowLen:-windowLen + 1]


def selfSimilarityMatrix(featureVectors):
    '''
    This function computes the self-similarity matrix for a sequence of feature vectors.
    ARGUMENTS:
     - featureVectors:     a np matrix (nDims x nVectors) whose i-th column corresponds to the i-th feature vector

    RETURNS:
     - S:             the self-similarity matrix (nVectors x nVectors)
    '''

    [nDims, nVectors] = featureVectors.shape
    [featureVectors2, MEAN, STD] = aT.normalizeFeatures([featureVectors.T])
    featureVectors2 = featureVectors2[0].T
    S = 1.0 - distance.squareform(distance.pdist(featureVectors2.T, 'cosine'))
    return S


def flags2segs(Flags, window):
    '''
    ARGUMENTS:
     - Flags:     a sequence of class flags (per time window)
     - window:    window duration (in seconds)

    RETURNS:
     - segs:    a sequence of segment's limits: segs[i,0] is start and segs[i,1] are start and end point of segment i
     - classes:    a sequence of class flags: class[i] is the class ID of the i-th segment
    '''

    preFlag = 0
    curFlag = 0
    numOfSegments = 0

    curVal = Flags[curFlag]
    segsList = []
    classes = []
    while (curFlag < len(Flags) - 1):
        stop = 0
        preFlag = curFlag
        preVal = curVal
        while (stop == 0):
            curFlag = curFlag + 1
            tempVal = Flags[curFlag]
            if ((tempVal != curVal) | (curFlag == len(Flags) - 1)):  # stop
                numOfSegments = numOfSegments + 1
                stop = 1
                curSegment = curVal
                curVal = Flags[curFlag]
                segsList.append((curFlag * window))
                classes.append(preVal)
    segs = np.zeros((len(segsList), 2))

    for i in range(len(segsList)):
        if i > 0:
            segs[i, 0] = segsList[i - 1]
        segs[i, 1] = segsList[i]
    return (segs, classes)


def segs2flags(segStart, segEnd, segLabel, winSize):
    '''
    This function converts segment endpoints and respective segment labels to fix-sized class labels.
    ARGUMENTS:
     - segStart:    segment start points (in seconds)
     - segEnd:    segment endpoints (in seconds)
     - segLabel:    segment labels
      - winSize:    fix-sized window (in seconds)
    RETURNS:
     - flags:    np array of class indices
     - classNames:    list of classnames (strings)
    '''
    flags = []
    classNames = list(set(segLabel))
    curPos = winSize / 2.0
    while curPos < segEnd[-1]:
        for i in range(len(segStart)):
            if curPos > segStart[i] and curPos <= segEnd[i]:
                break
        flags.append(classNames.index(segLabel[i]))
        curPos += winSize
    return np.array(flags), classNames


def computePreRec(CM, classNames):
    '''
    This function computes the Precision, Recall and F1 measures, given a confusion matrix
    '''
    numOfClasses = CM.shape[0]
    if len(classNames) != numOfClasses:
        print("Error in computePreRec! Confusion matrix and classNames list must be of the same size!")
        return
    Precision = []
    Recall = []
    F1 = []
    for i, c in enumerate(classNames):
        Precision.append(CM[i, i] / np.sum(CM[:, i]))
        Recall.append(CM[i, i] / np.sum(CM[i, :]))
        F1.append(2 * Precision[-1] * Recall[-1] / (Precision[-1] + Recall[-1]))
    return Recall, Precision, F1


def readSegmentGT(gtFile):
    '''
    This function reads a segmentation ground truth file, following a simple CSV format with the following columns:
    <segment start>,<segment end>,<class label>

    ARGUMENTS:
     - gtFile:       the path of the CSV segment file
    RETURNS:
     - segStart:     a numpy array of segments' start positions
     - segEnd:       a numpy array of segments' ending positions
     - segLabel:     a list of respective class labels (strings)
    '''
    f = open(gtFile, "r")
    print(gtFile)
    reader = csv.reader(f, delimiter=',')
    segStart = []
    segEnd = []
    segLabel = []
    for row in reader:
        if len(row) == 3:
            segStart.append(float(row[0]))
            segEnd.append(float(row[1]))
            # if row[2]!="other":
            #    segLabel.append((row[2]))
            # else:
            #    segLabel.append("silence")
            segLabel.append((row[2]))
    return np.array(segStart), np.array(segEnd), segLabel


def plotSegmentationResults(flagsInd, flagsIndGT, classNames, mtStep, ONLY_EVALUATE=False):
    '''
    This function plots statistics on the classification-segmentation results produced either by the fix-sized supervised method or the HMM method.
    It also computes the overall accuracy achieved by the respective method if ground-truth is available.
    '''
    flags = [classNames[int(f)] for f in flagsInd]
    (segs, classes) = flags2segs(flags, mtStep)
    minLength = min(flagsInd.shape[0], flagsIndGT.shape[0])
    if minLength > 0:
        accuracy = np.sum(flagsInd[0:minLength] == flagsIndGT[0:minLength]) / float(minLength)
    else:
        accuracy = -1

    if not ONLY_EVALUATE:
        Duration = segs[-1, 1]
        SPercentages = np.zeros((len(classNames), 1))
        Percentages = np.zeros((len(classNames), 1))
        AvDurations = np.zeros((len(classNames), 1))

        for iSeg in range(segs.shape[0]):
            SPercentages[classNames.index(classes[iSeg])] += (segs[iSeg, 1] - segs[iSeg, 0])

        for i in range(SPercentages.shape[0]):
            Percentages[i] = 100.0 * SPercentages[i] / Duration
            S = sum(1 for c in classes if c == classNames[i])
            if S > 0:
                AvDurations[i] = SPercentages[i] / S
            else:
                AvDurations[i] = 0.0

        for i in range(Percentages.shape[0]):
            print(classNames[i], Percentages[i], AvDurations[i])

        font = {'size': 10}
        plt.rc('font', **font)

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.set_yticks(np.array(range(len(classNames))))
        ax1.axis((0, Duration, -1, len(classNames)))
        ax1.set_yticklabels(classNames)
        ax1.plot(np.array(range(len(flagsInd))) * mtStep + mtStep / 2.0, flagsInd)
        if flagsIndGT.shape[0] > 0:
            ax1.plot(np.array(range(len(flagsIndGT))) * mtStep + mtStep / 2.0, flagsIndGT + 0.05, '--r')
        plt.xlabel("time (seconds)")
        if accuracy >= 0:
            plt.title('Accuracy = {0:.1f}%'.format(100.0 * accuracy))

        ax2 = fig.add_subplot(223)
        plt.title("Classes percentage durations")
        ax2.axis((0, len(classNames) + 1, 0, 100))
        ax2.set_xticks(np.array(range(len(classNames) + 1)))
        ax2.set_xticklabels([" "] + classNames)
        print(np.array(range(len(classNames))) + 0.5)
        Percentages = Percentages.reshape(-1)
        ax2.bar(np.array(range(len(classNames))) + 0.5, Percentages)

        ax3 = fig.add_subplot(224)
        plt.title("Segment average duration per class")
        ax3.axis((0, len(classNames) + 1, 0, AvDurations.max()))
        ax3.set_xticks(np.array(range(len(classNames) + 1)))
        ax3.set_xticklabels([" "] + classNames)
        AvDurations = AvDurations.reshape(-1)
        ax3.bar(np.array(range(len(classNames))) + 0.5, AvDurations)
        fig.tight_layout()
        plt.show()
    return accuracy


def evaluateSpeakerDiarization(flags, flagsGT):
    minLength = min(flags.shape[0], flagsGT.shape[0])
    flags = flags[0:minLength]
    flagsGT = flagsGT[0:minLength]

    uFlags = np.unique(flags)
    uFlagsGT = np.unique(flagsGT)

    # compute contigency table:
    cMatrix = np.zeros((uFlags.shape[0], uFlagsGT.shape[0]))
    for i in range(minLength):
        cMatrix[int(np.nonzero(uFlags == flags[i])[0]), int(np.nonzero(uFlagsGT == flagsGT[i])[0])] += 1.0

    Nc, Ns = cMatrix.shape
    N_s = np.sum(cMatrix, axis=0)
    N_c = np.sum(cMatrix, axis=1)
    N = np.sum(cMatrix)

    purityCluster = np.zeros((Nc,))
    puritySpeaker = np.zeros((Ns,))
    # compute cluster purity:
    for i in range(Nc):
        purityCluster[i] = np.max((cMatrix[i, :])) / (N_c[i])

    for j in range(Ns):
        puritySpeaker[j] = np.max((cMatrix[:, j])) / (N_s[j])

    purityClusterMean = np.sum(purityCluster * N_c) / N
    puritySpeakerMean = np.sum(puritySpeaker * N_s) / N

    return purityClusterMean, puritySpeakerMean


def trainHMM_computeStatistics(features, labels):
    '''
    This function computes the statistics used to train an HMM joint segmentation-classification model
    using a sequence of sequential features and respective labels

    ARGUMENTS:
     - features:    a numpy matrix of feature vectors (numOfDimensions x numOfWindows)
     - labels:    a numpy array of class indices (numOfWindows x 1)
    RETURNS:
     - startprob:    matrix of prior class probabilities (numOfClasses x 1)
     - transmat:    transition matrix (numOfClasses x numOfClasses)
     - means:    means matrix (numOfDimensions x 1)
     - cov:        deviation matrix (numOfDimensions x 1)
    '''
    uLabels = np.unique(labels)
    nComps = len(uLabels)

    nFeatures = features.shape[0]

    if features.shape[1] < labels.shape[0]:
        print("trainHMM warning: number of short-term feature vectors must be greater or equal to the labels length!")
        labels = labels[0:features.shape[1]]

    # compute prior probabilities:
    startprob = np.zeros((nComps,))
    for i, u in enumerate(uLabels):
        startprob[i] = np.count_nonzero(labels == u)
    startprob = startprob / startprob.sum()  # normalize prior probabilities

    # compute transition matrix:
    transmat = np.zeros((nComps, nComps))
    for i in range(labels.shape[0] - 1):
        transmat[int(labels[i]), int(labels[i + 1])] += 1
    for i in range(nComps):  # normalize rows of transition matrix:
        transmat[i, :] /= transmat[i, :].sum()

    means = np.zeros((nComps, nFeatures))
    for i in range(nComps):
        means[i, :] = np.matrix(features[:, np.nonzero(labels == uLabels[i])[0]].mean(axis=1))

    cov = np.zeros((nComps, nFeatures))
    for i in range(nComps):
        # cov[i,:,:] = np.cov(features[:,np.nonzero(labels==uLabels[i])[0]])  # use this lines if HMM using full gaussian distributions are to be used!
        cov[i, :] = np.std(features[:, np.nonzero(labels == uLabels[i])[0]], axis=1)

    return startprob, transmat, means, cov


def trainHMM_fromFile(wavFile, gtFile, hmmModelName, mtWin, mtStep):
    '''
    This function trains a HMM model for segmentation-classification using a single annotated audio file
    ARGUMENTS:
     - wavFile:        the path of the audio filename
     - gtFile:         the path of the ground truth filename
                       (a csv file of the form <segment start in seconds>,<segment end in seconds>,<segment label> in each row
     - hmmModelName:   the name of the HMM model to be stored
     - mtWin:          mid-term window size
     - mtStep:         mid-term window step
    RETURNS:
     - hmm:            an object to the resulting HMM
     - classNames:     a list of classNames

    After training, hmm, classNames, along with the mtWin and mtStep values are stored in the hmmModelName file
    '''

    [segStart, segEnd, segLabels] = readSegmentGT(gtFile)  # read ground truth data
    flags, classNames = segs2flags(segStart, segEnd, segLabels, mtStep)  # convert to fix-sized sequence of flags

    [Fs, x] = audio_basic_io.read_audio_file(wavFile)  # read audio data
    # F = aF.stFeatureExtraction(x, Fs, 0.050*Fs, 0.050*Fs);
    [F, _] = aF.mt_feature_extraction(x, Fs, mtWin * Fs, mtStep * Fs, round(Fs * 0.050),
                                      round(Fs * 0.050))  # feature extraction
    startprob, transmat, means, cov = trainHMM_computeStatistics(F,
                                                                 flags)  # compute HMM statistics (priors, transition matrix, etc)

    hmm = hmmlearn.hmm.GaussianHMM(startprob.shape[0], "diag")  # hmm training

    hmm.startprob_ = startprob
    hmm.transmat_ = transmat
    hmm.means_ = means
    hmm.covars_ = cov

    fo = open(hmmModelName, "wb")  # output to file
    cPickle.dump(hmm, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(classNames, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(mtWin, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(mtStep, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()

    return hmm, classNames


def trainHMM_fromDir(dirPath, hmmModelName, mtWin, mtStep):
    '''
    This function trains a HMM model for segmentation-classification using a where WAV files and .segment (ground-truth files) are stored
    ARGUMENTS:
     - dirPath:        the path of the data diretory
     - hmmModelName:    the name of the HMM model to be stored
     - mtWin:        mid-term window size
     - mtStep:        mid-term window step
    RETURNS:
     - hmm:            an object to the resulting HMM
     - classNames:        a list of classNames

    After training, hmm, classNames, along with the mtWin and mtStep values are stored in the hmmModelName file
    '''

    flagsAll = np.array([])
    classesAll = []
    for i, f in enumerate(glob.glob(dirPath + os.sep + '*.wav')):  # for each WAV file
        wavFile = f
        gtFile = f.replace('.wav', '.segments')  # open for annotated file
        if not os.path.isfile(gtFile):  # if current WAV file does not have annotation -> skip
            continue
        [segStart, segEnd, segLabels] = readSegmentGT(gtFile)  # read GT data
        flags, classNames = segs2flags(segStart, segEnd, segLabels, mtStep)  # convert to flags
        for c in classNames:  # update classnames:
            if c not in classesAll:
                classesAll.append(c)
        [Fs, x] = audio_basic_io.read_audio_file(wavFile)  # read audio data
        [F, _] = aF.mt_feature_extraction(x, Fs, mtWin * Fs, mtStep * Fs, round(Fs * 0.050),
                                          round(Fs * 0.050))  # feature extraction

        lenF = F.shape[1]
        lenL = len(flags)
        MIN = min(lenF, lenL)
        F = F[:, 0:MIN]
        flags = flags[0:MIN]

        flagsNew = []
        for j, fl in enumerate(flags):  # append features and labels
            flagsNew.append(classesAll.index(classNames[flags[j]]))

        flagsAll = np.append(flagsAll, np.array(flagsNew))

        if i == 0:
            Fall = F
        else:
            Fall = np.concatenate((Fall, F), axis=1)
    startprob, transmat, means, cov = trainHMM_computeStatistics(Fall, flagsAll)  # compute HMM statistics
    hmm = hmmlearn.hmm.GaussianHMM(startprob.shape[0], "diag")  # train HMM
    hmm.startprob_ = startprob
    hmm.transmat_ = transmat
    hmm.means_ = means
    hmm.covars_ = cov

    fo = open(hmmModelName, "wb")  # save HMM model
    cPickle.dump(hmm, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(classesAll, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(mtWin, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(mtStep, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()

    return hmm, classesAll


def hmmSegmentation(wavFileName, hmmModelName, PLOT=False, gtFileName=""):
    [Fs, x] = audio_basic_io.read_audio_file(wavFileName)  # read audio data

    try:
        fo = open(hmmModelName, "rb")
    except IOError:
        print("didn't find file")
        return

    try:
        hmm = cPickle.load(fo)
        classesAll = cPickle.load(fo)
        mtWin = cPickle.load(fo)
        mtStep = cPickle.load(fo)
    except:
        fo.close()
    fo.close()

    # Features = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.050*Fs);    # feature extraction
    [Features, _] = aF.mt_feature_extraction(x, Fs, mtWin * Fs, mtStep * Fs, round(Fs * 0.050), round(Fs * 0.050))
    flagsInd = hmm.predict(Features.T)  # apply model
    # for i in range(len(flagsInd)):
    #    if classesAll[flagsInd[i]]=="silence":
    #        flagsInd[i]=classesAll.index("speech")

    # plot results
    if os.path.isfile(gtFileName):
        [segStart, segEnd, segLabels] = readSegmentGT(gtFileName)
        flagsGT, classNamesGT = segs2flags(segStart, segEnd, segLabels, mtStep)
        flagsGTNew = []
        for j, fl in enumerate(flagsGT):  # "align" labels with GT
            if classNamesGT[flagsGT[j]] in classesAll:
                flagsGTNew.append(classesAll.index(classNamesGT[flagsGT[j]]))
            else:
                flagsGTNew.append(-1)
        CM = np.zeros((len(classNamesGT), len(classNamesGT)))
        flagsIndGT = np.array(flagsGTNew)
        for i in range(min(flagsInd.shape[0], flagsIndGT.shape[0])):
            CM[int(flagsIndGT[i]), int(flagsInd[i])] += 1
    else:
        flagsIndGT = np.array([])
    acc = plotSegmentationResults(flagsInd, flagsIndGT, classesAll, mtStep, not PLOT)
    if acc >= 0:
        print("Overall Accuracy: {0:.2f}".format(acc))
        return (flagsInd, classNamesGT, acc, CM)
    else:
        return (flagsInd, classesAll, -1, -1)


def mtFileClassification(inputFile, modelName, modelType, plotResults=False, gtFile=""):
    '''
    This function performs mid-term classification of an audio stream.
    Towards this end, supervised knowledge is used, i.e. a pre-trained classifier.
    ARGUMENTS:
        - inputFile:        path of the input WAV file
        - modelName:        name of the classification model
        - modelType:        svm or knn depending on the classifier type
        - plotResults:      True if results are to be plotted using matplotlib along with a set of statistics

    RETURNS:
          - segs:           a sequence of segment's endpoints: segs[i] is the endpoint of the i-th segment (in seconds)
          - classes:        a sequence of class flags: class[i] is the class ID of the i-th segment
    '''

    if not os.path.isfile(modelName):
        print("mtFileClassificationError: input modelType not found!")
        return (-1, -1, -1, -1)
    # Load classifier:
    if (modelType == 'svm') or (modelType == 'svm_rbf'):
        [Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, computeBEAT] = aT.loadSVModel(modelName)
    elif modelType == 'knn':
        [Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, computeBEAT] = aT.loadKNNModel(modelName)
    elif modelType == 'randomforest':
        [Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, computeBEAT] = aT.loadRandomForestModel(
            modelName)
    elif modelType == 'gradientboosting':
        [Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, computeBEAT] = aT.loadGradientBoostingModel(
            modelName)
    elif modelType == 'extratrees':
        [Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, computeBEAT] = aT.loadExtraTreesModel(
            modelName)

    if computeBEAT:
        print("Model " + modelName + " contains long-term music features (beat etc) and cannot be used in segmentation")
        return (-1, -1, -1, -1)
    [Fs, x] = audio_basic_io.read_audio_file(inputFile)  # load input file
    if Fs == -1:  # could not read file
        return (-1, -1, -1, -1)
    x = audio_basic_io.stereo2mono(x)  # convert stereo (if) to mono
    Duration = len(x) / Fs
    # mid-term feature extraction:
    [MidTermFeatures, _] = aF.mt_feature_extraction(x, Fs, mtWin * Fs, mtStep * Fs, round(Fs * stWin),
                                                    round(Fs * stStep))
    flags = []
    Ps = []
    flagsInd = []
    for i in range(MidTermFeatures.shape[1]):  # for each feature vector (i.e. for each fix-sized segment):
        curFV = (MidTermFeatures[:, i] - MEAN) / STD  # normalize current feature vector
        [Result, P] = aT.classifierWrapper(Classifier, modelType, curFV)  # classify vector
        flagsInd.append(Result)
        flags.append(classNames[int(Result)])  # update class label matrix
        Ps.append(np.max(P))  # update probability matrix
    flagsInd = np.array(flagsInd)

    # 1-window smoothing
    for i in range(1, len(flagsInd) - 1):
        if flagsInd[i - 1] == flagsInd[i + 1]:
            flagsInd[i] = flagsInd[i + 1]
    (segs, classes) = flags2segs(flags, mtStep)  # convert fix-sized flags to segments and classes
    segs[-1] = len(x) / float(Fs)

    # Load grount-truth:        
    if os.path.isfile(gtFile):
        [segStartGT, segEndGT, segLabelsGT] = readSegmentGT(gtFile)
        flagsGT, classNamesGT = segs2flags(segStartGT, segEndGT, segLabelsGT, mtStep)
        flagsIndGT = []
        for j, fl in enumerate(flagsGT):  # "align" labels with GT
            if classNamesGT[flagsGT[j]] in classNames:
                flagsIndGT.append(classNames.index(classNamesGT[flagsGT[j]]))
            else:
                flagsIndGT.append(-1)
        flagsIndGT = np.array(flagsIndGT)
        CM = np.zeros((len(classNamesGT), len(classNamesGT)))
        for i in range(min(flagsInd.shape[0], flagsIndGT.shape[0])):
            CM[int(flagsIndGT[i]), int(flagsInd[i])] += 1
    else:
        CM = []
        flagsIndGT = np.array([])
    acc = plotSegmentationResults(flagsInd, flagsIndGT, classNames, mtStep, not plotResults)
    if acc >= 0:
        print("Overall Accuracy: {0:.3f}".format(acc))
        return (flagsInd, classNamesGT, acc, CM)
    else:
        return (flagsInd, classNames, acc, CM)


def evaluateSegmentationClassificationDir(dirName, modelName, methodName):
    flagsAll = np.array([])
    classesAll = []
    accuracys = []

    for i, f in enumerate(glob.glob(dirName + os.sep + '*.wav')):  # for each WAV file
        wavFile = f
        print(wavFile)
        gtFile = f.replace('.wav', '.segments')  # open for annotated file

        if methodName.lower() in ["svm", "svm_rbf", "knn", "randomforest", "gradientboosting", "extratrees"]:
            flagsInd, classNames, acc, CMt = mtFileClassification(wavFile, modelName, methodName, False, gtFile)
        else:
            flagsInd, classNames, acc, CMt = hmmSegmentation(wavFile, modelName, False, gtFile)
        if acc > -1:
            if i == 0:
                CM = np.copy(CMt)
            else:
                CM = CM + CMt
            accuracys.append(acc)
            print(CMt, classNames)
            print(CM)
            [Rec, Pre, F1] = computePreRec(CMt, classNames)

    CM = CM / np.sum(CM)
    [Rec, Pre, F1] = computePreRec(CM, classNames)

    print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print("Average Accuracy: {0:.1f}".format(100.0 * np.array(accuracys).mean()))
    print("Average Recall: {0:.1f}".format(100.0 * np.array(Rec).mean()))
    print("Average Precision: {0:.1f}".format(100.0 * np.array(Pre).mean()))
    print("Average F1: {0:.1f}".format(100.0 * np.array(F1).mean()))
    print("Median Accuracy: {0:.1f}".format(100.0 * np.median(np.array(accuracys))))
    print("Min Accuracy: {0:.1f}".format(100.0 * np.array(accuracys).min()))
    print("Max Accuracy: {0:.1f}".format(100.0 * np.array(accuracys).max()))


def silenceRemoval(x, Fs, stWin, stStep, smoothWindow=0.5, Weight=0.5, plot=False):
    '''
    Event Detection (silence removal)
    ARGUMENTS:
         - x:                the input audio signal
         - Fs:               sampling freq
         - stWin, stStep:    window size and step in seconds
         - smoothWindow:     (optinal) smooth window (in seconds)
         - Weight:           (optinal) weight factor (0 < Weight < 1) the higher, the more strict
         - plot:             (optinal) True if results are to be plotted
    RETURNS:
         - segmentLimits:    list of segment limits in seconds (e.g [[0.1, 0.9], [1.4, 3.0]] means that
                    the resulting segments are (0.1 - 0.9) seconds and (1.4, 3.0) seconds
    '''

    if Weight >= 1:
        Weight = 0.99
    if Weight <= 0:
        Weight = 0.01

    # Step 1: feature extraction
    x = audio_basic_io.stereo2mono(x)  # convert to mono
    ShortTermFeatures = aF.st_feature_extraction(x, Fs, stWin * Fs, stStep * Fs)  # extract short-term features

    # Step 2: train binary SVM classifier of low vs high energy frames
    EnergySt = ShortTermFeatures[1, :]  # keep only the energy short-term sequence (2nd feature)
    E = np.sort(EnergySt)  # sort the energy feature values:
    L1 = int(len(E) / 10)  # number of 10% of the total short-term windows
    T1 = np.mean(E[0:L1]) + 0.000000000000001  # compute "lower" 10% energy threshold
    T2 = np.mean(E[-L1:-1]) + 0.000000000000001  # compute "higher" 10% energy threshold
    Class1 = ShortTermFeatures[:, np.where(EnergySt <= T1)[0]]  # get all features that correspond to low energy
    Class2 = ShortTermFeatures[:, np.where(EnergySt >= T2)[0]]  # get all features that correspond to high energy
    featuresSS = [Class1.T, Class2.T]  # form the binary classification task and ...

    [featuresNormSS, MEANSS, STDSS] = aT.normalizeFeatures(featuresSS)  # normalize and ...
    SVM = aT.trainSVM(featuresNormSS, 1.0)  # train the respective SVM probabilistic model (ONSET vs SILENCE)

    # Step 3: compute onset probability based on the trained SVM
    ProbOnset = []
    for i in range(ShortTermFeatures.shape[1]):  # for each frame
        curFV = (ShortTermFeatures[:, i] - MEANSS) / STDSS  # normalize feature vector
        ProbOnset.append(
            SVM.predict_proba(curFV.reshape(1, -1))[0][1])  # get SVM probability (that it belongs to the ONSET class)
    ProbOnset = np.array(ProbOnset)
    ProbOnset = smoothMovingAvg(ProbOnset, smoothWindow / stStep)  # smooth probability

    # Step 4A: detect onset frame indices:
    ProbOnsetSorted = np.sort(
        ProbOnset)  # find probability Threshold as a weighted average of top 10% and lower 10% of the values
    Nt = ProbOnsetSorted.shape[0] / 10
    Nt = int(Nt)
    T = (np.mean((1 - Weight) * ProbOnsetSorted[0:Nt]) + Weight * np.mean(ProbOnsetSorted[-Nt::]))

    MaxIdx = np.where(ProbOnset > T)[0]  # get the indices of the frames that satisfy the thresholding
    i = 0
    timeClusters = []
    segmentLimits = []

    # Step 4B: group frame indices to onset segments
    while i < len(MaxIdx):  # for each of the detected onset indices
        curCluster = [MaxIdx[i]]
        if i == len(MaxIdx) - 1:
            break
        while MaxIdx[i + 1] - curCluster[-1] <= 2:
            curCluster.append(MaxIdx[i + 1])
            i += 1
            if i == len(MaxIdx) - 1:
                break
        i += 1
        timeClusters.append(curCluster)
        segmentLimits.append([curCluster[0] * stStep, curCluster[-1] * stStep])

    # Step 5: Post process: remove very small segments:
    minDuration = 0.2
    segmentLimits2 = []
    for s in segmentLimits:
        if s[1] - s[0] > minDuration:
            segmentLimits2.append(s)
    segmentLimits = segmentLimits2

    if plot:
        timeX = np.arange(0, x.shape[0] / float(Fs), 1.0 / Fs)

        plt.subplot(2, 1, 1)
        plt.plot(timeX, x)
        for s in segmentLimits:
            plt.axvline(x=s[0])
            plt.axvline(x=s[1])
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(0, ProbOnset.shape[0] * stStep, stStep), ProbOnset)
        plt.title('Signal')
        for s in segmentLimits:
            plt.axvline(x=s[0])
            plt.axvline(x=s[1])
        plt.title('SVM Probability')
        plt.show()

    return segmentLimits


def speaker_diarization(file_name, num_speaker, mt_size=2.0,
                        mt_step=0.2, st_win=0.05, st_step=0.025,
                        lda_dim=35,
                        plot=False):
    '''
    ARGUMENTS:
        - fileName:        the name of the WAV file to be analyzed
        - numOfSpeakers    the number of speakers (clusters) in the recording (<=0 for unknown)
        - mtSize (opt)     mid-term window size
        - mtStep (opt)     mid-term window step
        - stWin  (opt)     short-term window size
        - LDAdim (opt)     LDA dimension (0 for no LDA)
        - PLOT     (opt)   0 for not plotting the results 1 for plottingy
    '''
    fr, x = audio_basic_io.read_audio_file(file_name)
    x = audio_basic_io.stereo2mono(x)
    duration = len(x) / fr

    classifier1, mean1, std1, class_names1, mt_win1, mt_step1, st_win1, st_step1, compute_beta1 = aT.loadKNNModel(
        os.path.join("data", "knnSpeakerAll"))
    classifier2, mean2, std2, class_names2, mt_win2, mt_step2, st_win2, st_step2, compute_beta2 = aT.loadKNNModel(
        os.path.join("data", "knnSpeakerFemaleMale"))

    mid_term_features, short_term_features = aF.mt_feature_extraction(signal=x,
                                                                      fr=fr,
                                                                      mt_win=mt_size * fr,
                                                                      mt_step=mt_step * fr,
                                                                      st_win=round(fr * st_win),
                                                                      st_step=round(fr * st_step))

    # (68, 329) (34, 2630)
    print(mid_term_features.shape, short_term_features.shape)
    mid_term_features2 = np.zeros((mid_term_features.shape[0] + len(class_names1) + len(class_names2),
                                   mid_term_features.shape[1]))

    for i in range(mid_term_features.shape[1]):
        cur_f1 = (mid_term_features[:, i] - mean1) / std1
        cur_f2 = (mid_term_features[:, i] - mean2) / std2
        result, p1 = aT.classifierWrapper(classifier1, "knn", cur_f1)
        result, p2 = aT.classifierWrapper(classifier2, "knn", cur_f2)
        mid_term_features2[0:mid_term_features.shape[0], i] = mid_term_features[:, i]
        mid_term_features2[mid_term_features.shape[0]:mid_term_features.shape[0] + len(class_names1), i] = p1 + 0.0001
        mid_term_features2[mid_term_features.shape[0] + len(class_names1)::, i] = p2 + 0.0001

    mid_term_features = mid_term_features2  # TODO
    # SELECT FEATURES:
    # iFeaturesSelect = [8,9,10,11,12,13,14,15,16,17,18,19,20];     # SET 0A
    # iFeaturesSelect = [8,9,10,11,12,13,14,15,16,17,18,19,20, 99,100];    # SET 0B
    # iFeaturesSelect = [8,9,10,11,12,13,14,15,16,17,18,19,20, 68,69,70,71,72,73,
    # 74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,
    # 97,98, 99,100];     # SET 0C

    i_features_select = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 41,
                         42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]  # SET 1A
    # iFeaturesSelect = [8,9,10,11,12,13,14,15,16,17,18,19,20,41,42,43,44,45,46,47,48,49,50,51,52,53, 99,100]; # SET 1B
    # iFeaturesSelect = [8,9,10,11,12,13,14,15,16,17,18,19,20,41,42,43,44,45,46,47,
    # 48,49,50,51,52,53, 68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,
    # 87,88,89,90,91,92,93,94,95,96,97,98, 99,100];     # SET 1C

    # iFeaturesSelect = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,34,35,
    # 36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53];  # SET 2A
    # iFeaturesSelect = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,34,35,
    # 36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53, 99,100];     # SET 2B
    # iFeaturesSelect = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,34,35,
    # 36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53, 68,69,70,71,72,73,74,75,
    # 76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98, 99,100];  # SET 2C

    # iFeaturesSelect = range(100);   # SET 3
    # MidTermFeatures += np.random.rand(MidTermFeatures.shape[0], MidTermFeatures.shape[1]) * 0.000000010

    mid_term_features = mid_term_features[i_features_select, :]

    mid_term_features_norm, mean, std = aT.normalizeFeatures([mid_term_features.T])
    mid_term_features_norm = mid_term_features_norm[0].T
    num_of_windows = mid_term_features.shape[1]

    # remove outliers:
    distances_all = np.sum(distance.squareform(distance.pdist(mid_term_features_norm.T)), axis=0)
    m_distances_all = np.mean(distances_all)
    i_non_out_liers = np.nonzero(distances_all < 1.2 * m_distances_all)[0]

    # TODO: Combine energy threshold for outlier removal:
    # EnergyMin = np.min(MidTermFeatures[1,:])
    # EnergyMean = np.mean(MidTermFeatures[1,:])
    # Thres = (1.5*EnergyMin + 0.5*EnergyMean) / 2.0
    # iNonOutLiers = np.nonzero(MidTermFeatures[1,:] > Thres)[0]
    # print(iNonOutLiers

    # per_out_lier = (100.0 * (num_of_windows - i_non_out_liers.shape[0])) / num_of_windows
    mid_term_features_norm_or = mid_term_features_norm
    mid_term_features_norm = mid_term_features_norm[:, i_non_out_liers]

    # LDA dimensionality reduction:
    if lda_dim > 0:
        mt_win_ratio = int(round(mt_size / st_win))
        mt_step_ratio = int(round(st_win / st_win))
        mt_features_to_reduce = []
        num_of_features = len(short_term_features)
        num_of_statistics = 2
        # for i in range(numOfStatistics * numOfFeatures + 1):
        for i in range(num_of_statistics * num_of_features):
            mt_features_to_reduce.append([])

        for i in range(num_of_features):  # for each of the short-term features:
            cur_pos = 0
            n = len(short_term_features[i])
            while cur_pos < n:
                n1 = cur_pos
                n2 = cur_pos + mt_win_ratio
                if n2 > n:
                    n2 = n
                cur_st_features = short_term_features[i][n1:n2]
                mt_features_to_reduce[i].append(np.mean(cur_st_features))
                mt_features_to_reduce[i + num_of_features].append(np.std(cur_st_features))
                cur_pos += mt_step_ratio
        mt_features_to_reduce = np.array(mt_features_to_reduce)
        mt_features_to_reduce2 = np.zeros((mt_features_to_reduce.shape[0] + len(class_names1) + len(class_names2),
                                           mt_features_to_reduce.shape[1]))
        for i in range(mt_features_to_reduce.shape[1]):
            cur_f1 = (mt_features_to_reduce[:, i] - mean1) / std1
            cur_f2 = (mt_features_to_reduce[:, i] - mean2) / std2
            result, p1 = aT.classifierWrapper(classifier1, "knn", cur_f1)
            result, p2 = aT.classifierWrapper(classifier2, "knn", cur_f2)
            mt_features_to_reduce2[0:mt_features_to_reduce.shape[0], i] = mt_features_to_reduce[:, i]
            mt_features_to_reduce2[mt_features_to_reduce.shape[0]:mt_features_to_reduce.shape[0] + len(class_names1),
            i] = p1 + 0.0001
            mt_features_to_reduce2[mt_features_to_reduce.shape[0] + len(class_names1)::, i] = p2 + 0.0001
        mt_features_to_reduce = mt_features_to_reduce2
        mt_features_to_reduce = mt_features_to_reduce[i_features_select, :]
        # mtFeaturesToReduce += np.random.rand(mtFeaturesToReduce.shape[0], mtFeaturesToReduce.shape[1]) * 0.0000010
        mt_features_to_reduce, mean, std = aT.normalizeFeatures([mt_features_to_reduce.T])
        mt_features_to_reduce = mt_features_to_reduce[0].T
        # DistancesAll = np.sum(distance.squareform(distance.pdist(mtFeaturesToReduce.T)), axis=0)
        # MDistancesAll = np.mean(DistancesAll)
        # iNonOutLiers2 = np.nonzero(DistancesAll < 3.0*MDistancesAll)[0]
        # mtFeaturesToReduce = mtFeaturesToReduce[:, iNonOutLiers2]
        labels = np.zeros((mt_features_to_reduce.shape[1],))
        lda_step = 1.0
        lda_step_ratio = lda_step / st_win
        # print(LDAstep, LDAstepRatio
        for i in range(labels.shape[0]):
            labels[i] = int(i * st_win / lda_step_ratio)
        clf = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=lda_dim)
        clf.fit(mt_features_to_reduce.T, labels)
        mid_term_features_norm = (clf.transform(mid_term_features_norm.T)).T

    if num_speaker <= 0:
        s_range = range(2, 10)
    else:
        s_range = [num_speaker]
    cls_all = []
    sil_all = []
    centers_all = []
    # (26, 314)
    print('mid_term_features_norm', mid_term_features_norm.shape)
    for i_speakers in s_range:
        k_means = sklearn.cluster.KMeans(n_clusters=i_speakers)
        k_means.fit(mid_term_features_norm.T)
        cls = k_means.labels_
        means = k_means.cluster_centers_

        # Y = distance.squareform(distance.pdist(MidTermFeaturesNorm.T))
        cls_all.append(cls)
        centers_all.append(means)
        sil_a = []
        sil_b = []
        for c in range(i_speakers):  # for each speaker (i.e. for each extracted cluster)
            cluster_percent = np.nonzero(cls == c)[0].shape[0] / float(len(cls))
            if cluster_percent < 0.020:
                sil_a.append(0.0)
                sil_b.append(0.0)
            else:
                mid_term_features_norm_temp = mid_term_features_norm[:, cls == c]  # get subset of feature vectors
                # compute average distance between samples that belong to the cluster (a values)
                yt = distance.pdist(mid_term_features_norm_temp.T)
                sil_a.append(np.mean(yt) * cluster_percent)
                sil_bs = []
                for c2 in range(i_speakers):  # compute distances from samples of other clusters
                    if c2 != c:
                        cluster_percent2 = np.nonzero(cls == c2)[0].shape[0] / float(len(cls))
                        mid_term_features_norm_temp2 = mid_term_features_norm[:, cls == c2]
                        yt = distance.cdist(mid_term_features_norm_temp.T, mid_term_features_norm_temp2.T)
                        sil_bs.append(np.mean(yt) * (cluster_percent + cluster_percent2) / 2.0)
                sil_bs = np.array(sil_bs)
                # ... and keep the minimum value (i.e. the distance from the "nearest" cluster)
                sil_b.append(min(sil_bs))
        sil_a = np.array(sil_a)
        sil_b = np.array(sil_b)
        sil = []
        for c in range(i_speakers):  # for each cluster (speaker)
            sil.append((sil_b[c] - sil_a[c]) / (max(sil_b[c], sil_a[c]) + 0.00001))  # compute silhouette
        sil_all.append(np.mean(sil))  # keep the AVERAGE SILLOUETTE

    # silAll = silAll * (1.0/(np.power(np.array(sRange),0.5)))
    imax = np.argmax(sil_all)  # position of the maximum sillouette value
    n_speakers_final = s_range[imax]  # optimal number of clusters

    # generate the final set of cluster labels
    # (important: need to retrieve the outlier windows:
    # this is achieved by giving them the value of their nearest non-outlier window)
    cls = np.zeros((num_of_windows,))
    for i in range(num_of_windows):
        j = np.argmin(np.abs(i - i_non_out_liers))
        cls[i] = cls_all[imax][j]

    # Post-process method 1: hmm smoothing
    for i in range(1):
        startprob, transmat, means, cov = trainHMM_computeStatistics(mid_term_features_norm_or, cls)
        hmm = hmmlearn.hmm.GaussianHMM(startprob.shape[0], "diag")  # hmm training
        hmm.startprob_ = startprob
        hmm.transmat_ = transmat
        hmm.means_ = means
        hmm.covars_ = cov
        cls = hmm.predict(mid_term_features_norm_or.T)

        # Post-process method 2: median filtering:
    cls = scipy.signal.medfilt(cls, 13)
    cls = scipy.signal.medfilt(cls, 11)

    sil = sil_all[imax]  # final sillouette
    class_names = ["speaker{0:d}".format(c) for c in range(n_speakers_final)]

    # load ground-truth if available
    gt_file = file_name.replace('.wav', '.segments')  # open for annotated file
    if os.path.isfile(gt_file):  # if groundturh exists
        seg_start, seg_end, seg_labels = readSegmentGT(gt_file)  # read GT data
        flags_gt, class_names_gt = segs2flags(seg_start, seg_end, seg_labels, mt_step)  # convert to flags

    x = np.arange(len(cls)) * mt_step + mt_step / 2.0
    if plot:
        fig = plt.figure()
        if num_speaker > 0:
            ax1 = fig.add_subplot(111)
        else:
            ax1 = fig.add_subplot(211)
        ax1.set_yticks(np.array(range(len(class_names))))
        ax1.axis((0, duration, -1, len(class_names)))
        ax1.set_yticklabels(class_names)
        ax1.plot(x, cls)

    if os.path.isfile(gt_file):
        if plot:
            ax1.plot(np.array(range(len(flags_gt))) * mt_step + mt_step / 2.0, flags_gt, 'r')
        purity_cluster_mean, purity_speaker_mean = evaluateSpeakerDiarization(cls, flags_gt)
        print("{0:.1f}\t{1:.1f}".format(100 * purity_cluster_mean, 100 * purity_speaker_mean))
        if plot:
            plt.title("Cluster purity: {0:.1f}% - Speaker purity: {1:.1f}%".format(100 * purity_cluster_mean,
                                                                                   100 * purity_speaker_mean))
    if plot:
        plt.xlabel("time (seconds)")
        # print(sRange, silAll)
        if num_speaker <= 0:
            plt.subplot(212)
            plt.plot(s_range, sil_all)
            plt.xlabel("number of clusters")
            plt.ylabel("average clustering's sillouette")
        plt.show()
    return x, cls


def speakerDiarizationEvaluateScript(folderName, LDAs):
    '''
        This function prints the cluster purity and speaker purity for each WAV file stored in a provided directory (.SEGMENT files are needed as ground-truth)
        ARGUMENTS:
            - folderName:     the full path of the folder where the WAV and SEGMENT (ground-truth) files are stored
            - LDAs:            a list of LDA dimensions (0 for no LDA)
    '''
    types = ('*.wav',)
    wavFilesList = []
    for files in types:
        wavFilesList.extend(glob.glob(os.path.join(folderName, files)))

    wavFilesList = sorted(wavFilesList)

    # get number of unique speakers per file (from ground-truth)    
    N = []
    for wavFile in wavFilesList:
        gtFile = wavFile.replace('.wav', '.segments');
        if os.path.isfile(gtFile):
            [segStart, segEnd, segLabels] = readSegmentGT(gtFile)  # read GT data
            N.append(len(list(set(segLabels))))
        else:
            N.append(-1)

    for l in LDAs:
        print("LDA = {0:d}".format(l))
        for i, wavFile in enumerate(wavFilesList):
            speaker_diarization(wavFile, N[i], 2.0, 0.2, 0.05, l, plot=False)
        print


def musicThumbnailing(x, Fs, shortTermSize=1.0, shortTermStep=0.5, thumbnailSize=10.0, Limit1=0, Limit2=1):
    '''
    This function detects instances of the most representative part of a music recording, also called "music thumbnails".
    A technique similar to the one proposed in [1], however a wider set of audio features is used instead of chroma features.
    In particular the following steps are followed:
     - Extract short-term audio features. Typical short-term window size: 1 second
     - Compute the self-silimarity matrix, i.e. all pairwise similarities between feature vectors
      - Apply a diagonal mask is as a moving average filter on the values of the self-similarty matrix. 
       The size of the mask is equal to the desirable thumbnail length.
      - Find the position of the maximum value of the new (filtered) self-similarity matrix.
       The audio segments that correspond to the diagonial around that position are the selected thumbnails
    

    ARGUMENTS:
     - x:            input signal
     - Fs:            sampling frequency
     - shortTermSize:     window size (in seconds)
     - shortTermStep:    window step (in seconds)
     - thumbnailSize:    desider thumbnail size (in seconds)
    
    RETURNS:
     - A1:            beginning of 1st thumbnail (in seconds)
     - A2:            ending of 1st thumbnail (in seconds)
     - B1:            beginning of 2nd thumbnail (in seconds)
     - B2:            ending of 2nd thumbnail (in seconds)

    USAGE EXAMPLE:
       import audioFeatureExtraction as aF
     [Fs, x] = basicIO.readAudioFile(inputFile)
     [A1, A2, B1, B2] = musicThumbnailing(x, Fs)

    [1] Bartsch, M. A., & Wakefield, G. H. (2005). Audio thumbnailing of popular music using chroma-based representations. 
    Multimedia, IEEE Transactions on, 7(1), 96-104.
    '''
    x = audio_basic_io.stereo2mono(x);
    # feature extraction:
    stFeatures = aF.st_feature_extraction(x, Fs, Fs * shortTermSize, Fs * shortTermStep)

    # self-similarity matrix
    S = selfSimilarityMatrix(stFeatures)

    # moving filter:
    M = int(round(thumbnailSize / shortTermStep))
    B = np.eye(M, M)
    S = scipy.signal.convolve2d(S, B, 'valid')

    # post-processing (remove main diagonal elements)
    MIN = np.min(S)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if abs(i - j) < 5.0 / shortTermStep or i > j:
                S[i, j] = MIN;

    # find max position:
    S[0:int(Limit1 * S.shape[0]), :] = MIN
    S[:, 0:int(Limit1 * S.shape[0])] = MIN
    S[int(Limit2 * S.shape[0])::, :] = MIN
    S[:, int(Limit2 * S.shape[0])::] = MIN

    maxVal = np.max(S)
    [I, J] = np.unravel_index(S.argmax(), S.shape)
    # plt.imshow(S)
    # plt.show()
    # expand:
    i1 = I;
    i2 = I
    j1 = J;
    j2 = J

    while i2 - i1 < M:
        if i1 <= 0 or j1 <= 0 or i2 >= S.shape[0] - 2 or j2 >= S.shape[1] - 2:
            break
        if S[i1 - 1, j1 - 1] > S[i2 + 1, j2 + 1]:
            i1 -= 1
            j1 -= 1
        else:
            i2 += 1
            j2 += 1

    return (shortTermStep * i1, shortTermStep * i2, shortTermStep * j1, shortTermStep * j2, S)
