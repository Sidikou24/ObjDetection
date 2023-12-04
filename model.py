import tensorflow as tf
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Creation du model
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(224, 224,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#Compilation du modèle
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#pretraitement et augmentation des données
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
training_set = train_datagen.flow_from_directory('/jeu_de_données_images/Train', target_size = (224,224), batch_size = 16, class_mode = 'binary')

#Entrainement du model
model.fit(training_set, epochs=10)

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('/jeu_de_données_images/Test', target_size=(224, 224), batch_size=32, class_mode='binary')


model.save('model_incendie.h5')
