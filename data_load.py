# -*- coding: utf-8 -*-
# /usr/bin/python2

import glob
import random

import librosa
import numpy as np
from tensorpack.dataflow.base import RNGDataFlow
from tensorpack.dataflow.common import BatchData
from tensorpack.dataflow import PrefetchData
from audio import read_wav, preemphasis, amp2db
from hparam import hparam
from utils import normalize_0_1
hp = hparam()

'''
class Net1DataFlow(DataFlow):
    def get_data(self):
        while True:
            npz_file = random.choice(self.npz_files)
            yield read_mfccs_and_phones(npz_file)
    def size(self):
        return len(self.npz_files)
'''
def wav_random_crop(wav, sr, duration):
    assert (wav.ndim <= 2)

    target_len = sr * duration
    wav_len = wav.shape[-1]
    start = np.random.choice(range(np.maximum(1, wav_len - target_len)), 1)[0]
    end = start + target_len
    if wav.ndim == 1:
        wav = wav[start:end]
    else:
        wav = wav[:, start:end]
    return wav


def get_mfccs_and_phones(wav_file, trim=False, random_shuffle=False):

    '''This is applied in `train1` or `test1` phase.
    '''

    # Load
    wav = read_wav(wav_file, sr=hp.default.sr)

    mfccs, _, _ = _get_mfcc_and_spec(wav, hp.default.preemphasis, hp.default.n_fft,
                                     hp.default.win_length,
                                     hp.default.hop_length)

    # timesteps
    num_timesteps = mfccs.shape[0]

    # phones (targets)
    phn_file = wav_file.replace("WAV", "PHN")
    phn2idx, idx2phn = load_vocab()
    phns = np.zeros(shape=(num_timesteps,))
    bnd_list = []
    for line in open(phn_file, 'r').read().splitlines():
        start_point, _, phn = line.split()
        bnd = int(start_point) // hp.default.hop_length
        phns[bnd:] = phn2idx[phn]
        bnd_list.append(bnd)

    # Trim
    if trim:
        start, end = bnd_list[1], bnd_list[-1]
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    if random_shuffle:
        for i in range(len(bnd_list) - 1):
            start = bnd_list[i]
            end = bnd_list[i+1] - 1
            np.random.shuffle(mfccs[start:end])

    return mfccs, phns


def read_mfccs_and_phones(npz_file):
    np_arrays = np.load(npz_file)
    mfccs = np_arrays['mfccs']
    phns = np_arrays['phns']
    np_arrays.close()

    return mfccs, phns


def get_mfccs_and_spectrogram(wav_file, trim=False, random_crop=False, isConverting=False):
    '''This is applied in `train2`, `test2` or `convert` phase.
    '''


    # Load
    wav = read_wav(wav_file, sr=hp.default.sr)

    # Trim
    if trim:
        wav, _ = librosa.effects.trim(wav, frame_length=hp.default.win_length, hop_length=hp.default.hop_length)

    if random_crop:
        wav = wav_random_crop(wav, hp.default.sr, hp.default.duration)


    # Padding or crop if not Converting
    if isConverting is False:
        length = int(hp.default.sr * hp.default.duration)
        wav = librosa.util.fix_length(wav, length)

    return _get_mfcc_and_spec(wav, hp.default.preemphasis, hp.default.n_fft, hp.default.win_length, hp.default.hop_length)


# TODO refactoring
def _get_mfcc_and_spec(wav, preemphasis_coeff, n_fft, win_length, hop_length):

    # Pre-emphasis
    y_preem = preemphasis(wav, coeff=preemphasis_coeff)

    # Get spectrogram and power energy
    D = librosa.stft(y=y_preem, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(D)
    mag_e = np.sum(mag ** 2, axis=0)
    mag_e = np.where(mag_e==0, np.finfo(float).eps, mag_e)

    # Get mel-spectrogram
    mel_basis = librosa.filters.mel(hp.default.sr, hp.default.n_fft, hp.default.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t) # mel spectrogram
    mel = np.where(mel==0, np.finfo(float).eps, mel)

    # Get mfccs, amp to db
    mag_db = amp2db(mag)
    mel_db = amp2db(mel)
    mfccs = np.dot(librosa.filters.dct(hp.default.n_mfcc, mel_db.shape[0]), mel_db)

    mfccs[0,:] = np.log(mag_e)

    
    """
    # Normalizing mfcc so that has zero mean and unit variance
    mfccs_norm = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    """

    # Normalization (0 ~ 1)
    mag_db = normalize_0_1(mag_db, hp.default.max_db, hp.default.min_db)
    mel_db = normalize_0_1(mel_db, hp.default.max_db, hp.default.min_db)

    return mfccs.T, mag_db.T, mel_db.T  # (t, n_mfccs), (t, 1+n_fft/2), (t, n_mels)


def read_mfccs_and_spectrogram(npz_file):
    np_arrays = np.load(npz_file)

    mfccs = np_arrays['mfccs']
    mag_db = np_arrays['mag_db']
    mel_db = np_arrays['mel_db']

    np_arrays.close()

    return mfccs, mag_db, mel_db

phns = ['h#', 'pcl', 'tcl', 'kcl', 'bcl', 'dcl', 'gcl', 'pau', 'epi', # 0
        'aa','ao', #1
        'ah','ax','ax-h',#2
        'er','axr',#3
        'hh','hv',#4
        'ih','ix',#5
        'l','el',#6
        'm','em',#7
        'n','en','nx',#8
        'ng','eng',#9
        'sh','zh',#10
        'uw','ux',#11
        'iy','eh','ae','uh','ey','ay','oy','aw','ow','r','y','w','ch',
        'jh','dh','b','d','dx','g','p','t','k','z','v','f','th','s','q']

phns_map = [0,0,0,0,0,0,0,0,0,
            1,1,
            2,2,2,
            3,3,
            4,4,
            5,5,
            6,6,
            7,7,
            8,8,8,
            9,9,
            10,10,
            11,11,
            12,13,14,15,16,17,18,19,20,21,22,23,24,
            25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]


def load_vocab():
    phn2idx = {phn: idx for idx, phn in enumerate(phns)}
    idx2phn = {idx: phn for idx, phn in enumerate(phns)}

    return phn2idx, idx2phn
