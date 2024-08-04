import os
import librosa as lb
import random
import numpy as np
from config import *
from sklearn.model_selection import train_test_split
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from pydub import AudioSegment

def predict(file_names, labels, class_list, typeset, model):
    """
    Predict samples in a set (val/test set) (just .wav)
    Input: 
    - file_name: files directory (got from val(test)_generator.filenames)
    - labels: labels respective to file_name (got from val(test)_generator.labels)
    - class_list: Global variables 'class_list'
    - typeset: "val" or "test"
    - model: model used to predict

    Output:
    - y_pred_index: List of index class prediction of all samples in file_name
    - y_class_pred: List of class respective to y_pred_index
    """

    y_pred_index = []
    y_class_pred = []
    for file in file_names:
        file_root = DATASET_ROOT + "\\" + typeset + "\\" + str(file)
        image = load_img(file_root, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1], 3))
        image_array = img_to_array(image)
        image_array = image_array * 1./255
        input_data = tf.expand_dims(image_array, 0)
        pred = model.predict(input_data, verbose = 0)
        pred_index = np.argmax(np.squeeze(pred))
        y_pred_index.append(pred_index)
        y_class_pred.append(class_list[pred_index])
    print("---Predicted----")
    print("Accuracy on {} set : {}".format(typeset, (labels == y_pred_index).sum()/ len(labels)))
    return y_pred_index, y_class_pred

def predict_new30s(audio_dir, model, save_dir = TEST_IMAGES_ROOT):
    """
    Predict new 30s-length audio (arround 30 is accepted)
    Input:
    - audio_dir : List of audios directory (.wav)
    - model: model to predict
    - save_dir: TEST_IMAGES_ROOT - directory save log-mel-spec image of new audio

    Output:
    - y_pred: List of index class prediction of all samples in file_name
    - y_class: List of class respective to y_pred_index
    """

    y_pred = []
    y_class = []

    for dir in audio_dir:
        load_dir, sr = lb.load(dir)
        S = lb.feature.melspectrogram(y = load_dir, sr=sr)
        S_db = lb.amplitude_to_db(S, ref=np.max)

        # Create TEST_IMAGE_ROOT if it does not exist yey
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        audio_file_name = dir.split("\\")[-1][:-4]

        saved_img_root = save_dir + "\\{}".format(audio_file_name) + ".png"
        plt.imsave(saved_img_root, S_db)

        image = load_img(saved_img_root, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1], 3))
        image_array = img_to_array(image)
        input_data = tf.expand_dims(image_array, 0)
        pred = model.predict(input_data, verbose = 0)
        pred_index = np.argmax(np.squeeze(pred))
        y_pred.append(pred_index)
        y_class.append(class_list[pred_index])

        return y_pred, y_class
    


def predict_new(audio_dir, src_folder, model, save_dir, unit_length = 661500):
    """
    Predict audio of any length using one model
    Split each audio into several equal samples which length = unit_length (661500 = 30s), then feed to NN
    Get predict class by votting each sample's prediction

    Input:
    - audio_dir: List of audio directory to predict
    - src_folder: Dir of folder containning audio_dir
    - model: Model to predict
    - save_dir: Directory to save log-mel-spec image of samples splitted from each audio in audio_dir
    Output:
    - y_pred_index: List of index predicted of each audio in audio_dir
    - y_pred_class: Respective class predicted of y_pred_index
    """

    def mp3_2_wav(dir, dst, sample_rate = SR):
        """
        Convert mp3 to wav and save wav file to dst
        Input: dir (mp3)
        """
        # convert wav to mp3.
        sound = AudioSegment.from_mp3(dir)
        sound.set_frame_rate(sample_rate)
        sound.export(dst, format="wav")


    def process(samples_split, save_dir, file_name, is_saved):
        """
        End to end processing steps of each audio

        Input:
        - samples_split: List of samples splitted from each audio in audio_dir
        - save_dir: Directory to save log-mel-spec image of samples splitted from each audio in audio_dir
        - is_save: If False, do not save log-mel-spec image of samples, just make prediction

        Output:
        - np.array(samples_db_list): A batch of samples of each audio file (nums_of_sample, input_shape[0], input_shape[1], 3) to feed to NN
        """
        
        samples_db_list = []
        for i, sample in enumerate(samples_split):
            S = lb.feature.melspectrogram(y = sample, sr=sr)
            S_db = lb.amplitude_to_db(S, ref=np.max)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            sample_root = save_dir + "\\{}".format(file_name) + "_sample{}".format(i) + ".png"
            plt.imsave(sample_root, S_db)
            image = load_img(sample_root, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1], 3))
            img_array = img_to_array(image)
            img_array = img_array * 1./ 255
            samples_db_list.append(img_array)
            if not is_saved: # Not save mode
                for file in os.listdir(save_dir):
                    if file.endswith('.png'):
                        os.remove(save_dir + '\\' + file)
        return np.array(samples_db_list)

    # Define result
    y_pred_index = []
    y_pred_class = []

    # List of samples of each audio
    samples_split = []
    y_pred_split = []

    for dir in audio_dir:
        if dir.endswith(".mp3"):
            print("Convert {} to .wav".format(dir))
            wav_dir = src_folder + "\\" + dir.split("\\")[-1][:-4] + ".wav"    # src_folder = TEST_AUDIO_PATH (trainning), AUDIO_FROM_USER (web)
            mp3_2_wav(dir, wav_dir)
            dir = wav_dir       # Take wav dir for sampling
        audio, sr = lb.load(dir)
        if (len(audio) >= unit_length):
            # Number of sample of each audio
            nums_of_samples = len(audio) // unit_length
        else:
            err = "Audio length must be greater than 30s"
            print(err)
            return err
        for i in range(0, nums_of_samples):
            curr_sample = audio[i * unit_length : i * unit_length + unit_length]
            if (len(curr_sample) != unit_length): # Cannot sampling this curr_sample
                break
            samples_split.append(audio[i * unit_length : i * unit_length + unit_length])

        file_name = dir.split("\\")[-1][:-4]

        input_data = process(samples_split, save_dir, file_name, False)

        pred_candidates = model.predict(input_data, verbose = 0)

        pred_index_candidates = [np.argmax(sample) for sample in pred_candidates]

        pred_index = max(pred_index_candidates, key = pred_index_candidates.count)
        pred_class = class_list[pred_index]

        y_pred_index.append(pred_index)
        y_pred_class.append(pred_class)

        # Reset samples_split after passing one dir of audio_dir
        samples_split = []

    return y_pred_index, y_pred_class



def PROD_predict(audio_dir, src_folder, save_dir, model1, model2, model3, unit_length = 661500):
    """
    Predict audio of any length using PROD fusion of three models predicted probability vectors
    Split each audio into several equal sample which length = unit_length, then feed to NN
    Get predict class by most votting of each sample's prediction

    Input:
    - audio_dir: List of audio directory to predict
    - src_folder: Dir of folder containning audio_dir
    - model: Model to predict
    - save_dir: Directory to save log-mel-spec image of samples splitted from each audio in audio_dir
    Output:
    - y_pred_index: List of index predicted of each audio in audio_dir
    - y_pred_class: Respective class predicted of y_pred_index
    """

    def mp3_2_wav(dir, dst, sample_rate = 22050):
        """
        Convert mp3 to wav and save wav file to dst
        Input: dir (mp3)
        """
        # convert wav to mp3.
        sound = AudioSegment.from_mp3(dir)
        sound.set_frame_rate(sample_rate)
        sound.export(dst, format="wav")


    def process(samples_split, save_dir, file_name, is_saved):
        """
        End to end processing steps of each audio

        Input:
        - samples_split: List of samples splitted from each audio in audio_dir
        - save_dir: Directory to save log-mel-spec image of samples splitted from each audio in audio_dir
        - is_save: If False, do not save log-mel-spec image of samples, just make prediction

        Output:
        - np.array(samples_db_list): A batch of samples of each audio file (nums_of_sample, input_shape[0], input_shape[1], 3) to feed to NN
        """
        samples_db_list = []
        for i, sample in enumerate(samples_split):
            S = lb.feature.melspectrogram(y = sample, sr=sr)
            S_db = lb.amplitude_to_db(S, ref=np.max)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            sample_root = save_dir + "\\{}".format(file_name) + "_sample{}".format(i) + ".png"
            plt.imsave(sample_root, S_db)
            image = load_img(sample_root, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1], 3))
            img_array = img_to_array(image)
            img_array = img_array / 255
            samples_db_list.append(img_array)
            if not is_saved: # Not save mode
                for file in os.listdir(save_dir):
                    if file.endswith('.png'):
                        os.remove(save_dir + "\\" + file)
        return np.array(samples_db_list)

    # Define result
    y_pred_index = []
    y_pred_class = []

    # List of samples of each audio
    samples_split = []
    y_pred_split = []

    for dir in audio_dir:
        if dir.endswith(".mp3"):
            # Get file name
            wav_dir = src_folder + "\\" + dir.split("\\")[-1][:-4] + ".wav"
            mp3_2_wav(dir, wav_dir)
            dir = wav_dir       # Take wav dir for sampling
        print(dir)
        audio, sr = lb.load(dir)
        if (len(audio) >= unit_length):
            # Number of sample of each audio
            nums_of_samples = len(audio) // unit_length
        else:
            err = "Audio length must be greater than 30s"
            print(err)
            return err
        for i in range(0, nums_of_samples):
            curr_sample = audio[i * unit_length : i * unit_length + unit_length]
            if (len(curr_sample) != unit_length): # Cannot sampling this curr_sample
                break
            samples_split.append(audio[i * unit_length : i * unit_length + unit_length])

        file_name = dir.split("\\")[-1][:-4]

        input_data = process(samples_split, save_dir, file_name, False)

        pred_candidates1 = model1.predict(input_data, verbose = 0)

        pred_candidates2 = model2.predict(input_data, verbose = 0)

        pred_candidates3 = model3.predict(input_data, verbose = 0)

        PROD_probs = []

        # PROD fusion
        for i in range(pred_candidates1.shape[0]):
          PROD_probs.append(1/3 * pred_candidates1[i] * pred_candidates2[i] * pred_candidates3[i])

        pred_index_candidates = [np.argmax(sample) for sample in PROD_probs]

        pred_index = max(pred_index_candidates, key = pred_index_candidates.count)
        pred_class = class_list[pred_index]

        y_pred_index.append(pred_index)
        y_pred_class.append(pred_class)

        # Reset samples_split after passing one dir of audio_dir
        samples_split = []

    return y_pred_index, y_pred_class