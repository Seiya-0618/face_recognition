from deepface import DeepFace
import pandas as pd
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

class face_recognition():
    
    def __init__(self):
        print('load_img, profile, export_csv')

        
    
    def load_img(self, extension='jpeg'):
        #collect images by directory
        extension = str(extension)
        self.a = glob.glob("./data/img/*." + extension)
        return self.a
        
        
    def profile(self):
        flag=0
        #create dataframe
        emo_column = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        df_emotion = pd.DataFrame(columns = emo_column)
        age_series = pd.Series(name='age')
        gender_series = pd.Series(name='gender')
        for img in self.a:
            #convert images to array
            img2 = cv2.imread(img)
            #replace BGR to RGB
            img3 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            result = DeepFace.analyze(img3,actions=['emotion','age','gender'])
            #create bounding box
            #decide region
            region = result['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            #create mask
            mask = np.full((img3.shape[0], img3.shape[1]),False)
            #create line of bounding box
            mask[y:y+3, x: x+w] = True  # upper width
            mask[y+h:y+h+3, x: x+w] = True  # lower width
            mask[y: y+h, x:x+3] = True  # left height
            mask[y: y+h, x+w-3:x+w] = True  # right height
            #merge mask to img
            img3[mask==True] = 128
            #replace BGR to RGB
            img4 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
            judgement = cv2.imwrite('result/result_img/' + str(flag) + '.png', img4)            
            flag += 1
            print(judgement)
            emotion = result['emotion']
            emotion_list = []
            #create dataframe of emotion
            for i in emo_column:
                emotion_list.append(emotion[i])
            df_emotion.loc[img] = emotion_list
            age_series.loc[img] = result['age']
            gender_series.loc[img] = result['gender']
        #merge all dataframe
        output = pd.concat([df_emotion, age_series, gender_series],axis=1)
        output = output.reset_index(drop=True)
        self.output = output
        return output
    
    def export_csv(self):
        #export csv from profiled data
        self.output.to_csv('./result/result.csv')
