import os
import glob
import librosa
import numpy as np
import datetime
from numpy import savez
from hparam import hparam as hp
import pyworld as pw
from data_load import get_mfccs_and_phones
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

TIMIT_TRAIN_WAV = 'TIMIT/TRAIN/*/*/*.WAV'
TIMIT_TEST_WAV = 'TIMIT/TEST/*/*/*.WAV'

def generate_npz(index, wav_files, dataset_path, preproc_data_path):
    mfccs, phns = get_mfccs_and_phones(wav_files[index])
    f_name = wav_files[index].replace(dataset_path, preproc_data_path).replace('WAV', 'npz')

    if os.path.isfile(f_name):
        return
    else:
        savez(f_name, mfccs=mfccs, phns=phns)


def matching_list(wav_files, npz_files, dataset_path, preproc_data_path):
    print("Checking missing or corrupted files...")
    not_converted = []
    for i in tqdm(range(len(wav_files))):
        npz_name = wav_files[i].replace(dataset_path, preproc_data_path).replace('WAV','npz')
        if npz_name not in npz_files:
            not_converted.append(wav_files[i])

    for i in tqdm(range(len(npz_files))):
        wav_name = npz_files[i].replace(preproc_data_path, dataset_path).replace('npz', 'WAV')
        if wav_name not in wav_files:
            os.remove(npz_files[i])

    print("Done! {} files need to be converted!".format(len(not_converted)))
    return not_converted


def preprocessing(dataset_path, preproc_data_path, isConverting=False):
    s = datetime.datetime.now()

    train_wav_files = glob.glob(os.path.join(dataset_path, TIMIT_TRAIN_WAV))
    train_wav_dir = glob.glob(os.path.join(dataset_path, TIMIT_TRAIN_WAV.replace('*.WAV', '')))
    train_npz_files = glob.glob(os.path.join(preproc_data_path, TIMIT_TRAIN_WAV.replace('WAV', 'npz')))

    test_wav_files = glob.glob(os.path.join(dataset_path, TIMIT_TEST_WAV))
    test_wav_dir = glob.glob(os.path.join(dataset_path, TIMIT_TEST_WAV.replace('*.WAV', '')))
    test_npz_files = glob.glob(os.path.join(preproc_data_path, TIMIT_TEST_WAV.replace('WAV', 'npz')))

    for i in range(len(train_wav_dir)):
        if not os.path.isdir(train_wav_dir[i].replace(dataset_path, preproc_data_path)):
            os.makedirs(train_wav_dir[i].replace(dataset_path, preproc_data_path))

    for i in range(len(test_wav_dir)):
        if not os.path.isdir(test_wav_dir[i].replace(dataset_path, preproc_data_path)):
            os.makedirs(test_wav_dir[i].replace(dataset_path, preproc_data_path))

    pool = Pool(os.cpu_count())

    print('Starting pre-processing train dataset...')
    if len(train_npz_files) == 0:
        print("Extracting and saving features from wav files...")
        list(tqdm(pool.imap_unordered(partial(generate_npz, wav_files=train_wav_files,
                                              dataset_path=dataset_path, preproc_data_path=preproc_data_path),
                                      range(len(train_wav_files)), chunksize=1), total=len(train_wav_files)))
    else:
        convert_list = matching_list(train_wav_files, train_npz_files, dataset_path, preproc_data_path)
        if len(convert_list) == 0:
            print('All WAV files in train dataset are already converted!')
        else:
            print("Extracting and saving features from wav files...")
            list(tqdm(pool.imap_unordered(partial(generate_npz, wav_files=convert_list,
                                                  dataset_path=dataset_path, preproc_data_path=preproc_data_path),
                                          range(len(convert_list)), chunksize=1), total=len(convert_list)))
    print('Pre-processing of train dataset has finished!')

    print('Starting pre-processing test dataset...')
    if len(test_npz_files) == 0:
        print("Extracting and saving features from wav files...")
        list(tqdm(pool.imap_unordered(partial(generate_npz, wav_files=test_wav_files,
                                              dataset_path=dataset_path, preproc_data_path=preproc_data_path),
                                      range(len(test_wav_files)), chunksize=1), total=len(test_wav_files)))
    else:
        convert_list = matching_list(test_wav_files, test_npz_files, dataset_path, preproc_data_path)
        if len(convert_list) == 0:
            print('All WAV files in test dataset are already converted!')
        else:
            print("Extracting and saving features from wav files...")
            list(tqdm(pool.imap_unordered(partial(generate_npz, wav_files=convert_list,
                                                  dataset_path=dataset_path, preproc_data_path=preproc_data_path),
                                          range(len(convert_list)), chunksize=1), total=len(convert_list)))

    e = datetime.datetime.now()
    diff = e - s
    print("Done. elapsed time:{}s".format(diff.seconds))
    pool.close()
    pool.join()
