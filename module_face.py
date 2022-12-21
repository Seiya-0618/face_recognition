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
        emo_column = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        df_emotion = pd.DataFrame(columns = emo_column)
        age_series = pd.Series(name='age')
        gender_series = pd.Series(name='gender')
        for img in self.a:
            #convert images to array
            img2 = cv2.imread(img)
            #replace BGR to RGB
            img3 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            #plt.imshow(img3)
            result = DeepFace.analyze(img3,actions=['emotion','age','gender'])
            box = DeepFace.detectFace(img3, align=False)
            #face = cv2.cvtColor(box, cv2.COLOR_BGR2RGB)
            box = np.clip(box * 255, 0, 255).astype(np.uint8)
            tmp = cv2.imwrite('result/result_img/'+ str(flag) + '.png', box)
            flag += 1
            print(tmp)
            
            emotion = result['emotion']
            emotion_list = []
            #create dataframe of emotion
            for i in emo_column:
                emotion_list.append(emotion[i])
            df_emotion.loc[img] = emotion_list
            age_series.loc[img] = result['age']
            gender_series.loc[img] = result['gender']
        #merge all df
        output = pd.concat([df_emotion, age_series, gender_series],axis=1)
        output = output.reset_index(drop=True)
        self.output = output
        return output
    
    def export_csv(self):
        self.output.to_csv('./result/result.csv')
