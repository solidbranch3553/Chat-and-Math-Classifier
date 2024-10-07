import numpy as np
import json
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Embedding,LSTM,Bidirectional,GlobalMaxPooling1D,Conv1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

with open('intents.json') as j:
    intents = json.load(j)

texts = []
labels = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        texts.append(pattern.lower())
        labels.append(intent['tag'])

labelEncoding = LabelEncoder()
labelsEncoded = labelEncoding.fit_transform(labels)
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
x = pad_sequences(sequences, padding='post')
y = np.array(labelsEncoded)
pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))
pickle.dump(labelEncoding, open('labelEncoding.pkl', 'wb'))

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128))
model.add(Conv1D(filters=64, kernel_size=5,activation='relu',padding='same'))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(len(set(labelsEncoded)), activation='softmax', kernel_regularizer=l2(0.01)))
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.fit(x,y,epochs=100,batch_size=20,validation_split=0.2,verbose=1)
#model.save('text_C.h5')
#print("Model created!")

###TEST OUR MODEL
tLoad = pickle.load(open('tokenizer.pkl', 'rb'))
labelFile = pickle.load(open('labelEncoding.pkl', 'rb'))

def classify(inputU):
    sequ = tLoad.texts_to_sequences([inputU])
    paddedSequ = pad_sequences(sequ, maxlen=model.input_shape[1], padding='post')
    predictN = model.predict(paddedSequ)
    classIndex = np.argmax(predictN)
    label = labelFile.inverse_transform([classIndex])[0]
    return label

while True:
    userInput = input(">")
    result = classify(userInput)
    print(f"Classified as : {result}")