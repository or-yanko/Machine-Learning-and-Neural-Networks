# Imports
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tensorflow.keras.models.load_model("fruitRecognizorModel.h5")

try:
    model = tensorflow.keras.models.load_model("fruitRecognizorModel.h5")
except:
    #--------------------------------------------Data & model configuration---------------------------------------------
    # Data configuration
    training_set_folder = './Training_Smaller'
    test_set_folder = '' #===================================================--------------------------------------

    # Model configuration
    batch_size = 25
    img_width, img_height, img_num_channels = 25, 25, 3
    loss_function = sparse_categorical_crossentropy
    no_classes = 10
    no_epochs = 25
    optimizer = Adam()
    verbosity = 1


    #--------------------------------------------loading and preparing the data-----------------------------------------

    # Determine shape of the data
    input_shape = (img_width, img_height, img_num_channels)

    # Create a generator
    train_datagen = ImageDataGenerator(
      rescale=1./255
    )
    train_datagen = train_datagen.flow_from_directory(
            training_set_folder,
            save_to_dir='./adapted-images',
            save_format='jpeg',
            batch_size=batch_size,
            target_size=(25, 25),
            class_mode='sparse')

    #--------------------------------------------Specifying the model architecture--------------------------------------
    # Create the model
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(Conv2D(128, kernel_size=(5, 5), activation='relu'))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))

    # Display a model summary
    model.summary()


    #-------------------------------------Model compilation & starting the training process-----------------------------
    # Compile the model
    model.compile(loss=loss_function,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # Start training
    model.fit(
            train_datagen,
            epochs=no_epochs,
            shuffle=False)

    #-------------------------------------save modul with the name: fruitRecognizorModel -------------------------------
    model.save("fruitRecognizorModel.h5")


