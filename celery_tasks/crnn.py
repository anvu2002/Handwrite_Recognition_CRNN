import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import ctc_decode, get_value
import numpy as np
import pandas as pd
import cv2
from loguru import logger
import os
from os.path import join, dirname


class CRNN_Model:
    def __init__(self):
        model_path = join(dirname(__file__), "trained_models/CRNN_Handwrite_model.keras")
        self.model = load_model(model_path)
    
    def preprocess_img(self, img):
        (h, w) = img.shape
        
        final_img = np.ones([64, 256])*255 # blank white image
        
        # crop
        if w > 256:
            img = img[:, :256]
            
        if h > 64:
            img = img[:64, :]
        
        
        final_img[:h, :w] = img
        return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)
    
    def num_to_label(self, num):
        alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "

        ret = ""
        for ch in num:
            if ch == -1:  # CTC Blank
                break
            else:
                ret+=alphabets[ch]
        return ret

    def preprocess_input(self, img_path:str):
        img_set = []

        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        plt.imshow(image, cmap='gray')

        image = self.preprocess_img(image)
        image = image/255.
        img_set.append(image)

        return img_set
    
    def prepare_prediction(self, pred):
        # Decode the predictions
        decoded = ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True)[0][0]
        # Convert the decoded predictions to numpy array
        decoded = get_value(decoded)

        res = [ self.num_to_label(decoded[i]) for i in range(1)]

        return res

    def evaluate(img_list, n_img, predicted_result):
        y_true = img_list.loc[0:n_img, 'IDENTITY']
        correct_char = 0
        total_char = 0
        correct = 0

        true_result = []

        for i in range(n_img):
            pr = predicted_result[i]
            tr = y_true[i]
            total_char += len(tr)
            
            for j in range(min(len(tr), len(pr))):
                if tr[j] == pr[j]:
                    correct_char += 1
                    
            if pr == tr :
                correct += 1 
            
            true_result.append(tr)
        
        logger.info('Correct characters predicted : %.2f%%' %(correct_char*100/total_char))
        logger.info('Correct words predicted      : %.2f%%' %(correct*100/n_img))
        logger.info(f'\nTrue Results = {true_result}')


    def words_predict(self,img_path:str):
        try:
            logger.debug(f"[*] Processing Image: {img_path}")
           
            input_set = self.preprocess_input(img_path)

            pred = self.model.predict(np.array(input_set).reshape(-1, 256, 64, 1))
            
            result = self.prepare_prediction(pred)

            
            logger.debug(f"Predicted Handwriten Words = {result}")
            return {"predicted_words": result,"file_name" : img_path}
            
        except Exception as ex:
            logger.error(str(ex))
            return None