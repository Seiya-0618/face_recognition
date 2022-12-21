from deepface import DeepFace
import pandas as pd
import cv2
import glob
import matplotlib.pyplot as plt

class face_recognition():
    
    def __init__(self):
        print('create_dataframe　→load_img→profile:return output')

    def create_dataframe(self):
        self.emo_column = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.df_emotion = pd.DataFrame(columns = self.emo_column)
        self.age_series = pd.Series(name='age')
        self.gender_series = pd.Series(name='gender')
    
    def load_img(self):
        #collect images by directory
        self.a = glob.glob("./data/img/*.jpg")
        
        
    def profile(self):
        for img in self.a:
            #convert images to array
            img2 = cv2.imread(img)
            #replace BGR to RGB
            img3 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            result = DeepFace.analyze(img3,actions=['emotion','age','gender'])
            emotion = result['emotion']
            emotion_list = []
            #create dataframe of emotion
            for i in self.emo_column:
                emotion_list.append(emotion[i])
            self.df_emotion.loc[img] = emotion_list
            self.age_series.loc[img] = result['age']
            self.gender_series.loc[img] = result['gender']
        #merge all df
        output = pd.concat([self.df_emotion, self.age_series, self.gender_series],axis=1)
        self.output = output
        return output
    
    def export(self):
        self.output.to_csv('./result/result.csv')