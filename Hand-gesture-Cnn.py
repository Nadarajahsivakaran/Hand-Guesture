from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import warnings
warnings.filterwarnings("ignore")


classifier = Sequential()


classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,1),activation='relu',padding='same'))
classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,1),activation='relu',padding='same'))
classifier.add(MaxPooling2D(pool_size=(3,3),strides=2))
classifier.add(Dropout(0.5))


classifier.add(Convolution2D(64,(3,3),input_shape=(64,64,1),activation='relu',padding='same'))
classifier.add(Convolution2D(64,(3,3),input_shape=(64,64,1),activation='relu',padding='same'))
classifier.add(MaxPooling2D(pool_size=(3,3),strides=2))



classifier.add(Flatten())
classifier.add(Dense(units=64,activation='relu'))
classifier.add(Dense(units=52,activation='softmax'))


classifier.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
samplewise_center=True,
vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'data/train',
        target_size=(64,64),
        batch_size=208,
        color_mode='grayscale',
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'data/test',
        target_size=(64,64),
        batch_size=208,
        color_mode='grayscale',
        class_mode='categorical')


classifier.fit_generator(training_set,
        steps_per_epoch=7999,
        epochs=1,
        validation_data=test_set,
        validation_steps=4000)




classifier.save('model.h5')






