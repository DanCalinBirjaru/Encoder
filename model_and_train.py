import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import saved_model

# ----------------- FUNCTIONS -----------------

# BETTER SHORT TIME FOURIER TRANSFORM
def better_stft(data, fs = 1024, fft_size = 32, overlap_fac = 0.8):

    # fs = a scalar which is the sampling frequency of the data
    #fft_size = the size of the fast fourier transform window [power of 2]
    #overlap_fac = the overlap factor [0-1]


    hop_size = np.int32(np.floor(fft_size * (1-overlap_fac)))
    pad_end_size = fft_size          # the last segment can overlap the end of the data array by no more than one window size
    total_segments = np.int32(np.ceil(len(data) / np.float32(hop_size)))
    t_max = len(data) / np.float32(fs)

    window = np.hanning(fft_size)  # our half cosine window
    inner_pad = np.zeros(fft_size) # the zeros which will be used to double each segment size

    proc = np.concatenate((data, np.zeros(pad_end_size)))              # the data to process
    result = np.empty((total_segments, fft_size), dtype=np.float32)    # space to hold the result

    for i in range(total_segments):                      # for each segment
        current_hop = hop_size * i                        # figure out the current segment offset
        segment = proc[current_hop:current_hop+fft_size]  # get the current segment
        windowed = segment * window                       # multiply by the half cosine function
        padded = np.append(windowed, inner_pad)           # add 0s to double the length of the data
        spectrum = np.fft.fft(padded) / fft_size          # take the Fourier Transform and scale by the number of samples
        autopower = np.abs(spectrum * np.conj(spectrum))  # find the autopower spectrum
        result[i, :] = autopower[:fft_size]               # append to the results array

    #result = 20*np.log10(result)          # scale to db
    #result = np.clip(result,-85, -40)[:-(int(fft_size*0.15)),:].T   # clip values, and remove edge effects
    
    result = result[:-(int(fft_size*0.15)),:].T
    return result

# ----------------- DATA -----------------

mat = scipy.io.loadmat("Blip_strain.mat")
n = len(mat["outData"][:,0])

random_offset = np.random.randint(0, high = (int(n/4)-1) ) #-1 is for security to prevent calling out of bounds
    
wave_number = 3
data = mat["outData"][:,wave_number][random_offset:random_offset+int((3*n)/4)]

result = better_stft(data)

img_shape = result.shape
img_size = img_shape[0]*img_shape[1]

# ----------------- SPLIT TEST FROM TRAIN -----------------

test_train_ratio = 0.8

x_train = np.zeros([int(mat["outData"].shape[1]*test_train_ratio), result.shape[0]*result.shape[1]])
x_test = np.zeros([int(mat["outData"].shape[1]*(1-test_train_ratio))-1, result.shape[0]*result.shape[1]])

# ----------------- CONVERT TO STFT -----------------

for i in np.arange(len(x_train)):
    random_offset = np.random.randint(0, high = (int(n/4)-1) ) #-1 is for security to prevent calling out of bounds
    #random_offset = (int(n/8)-1)
    
    wave_ = mat["outData"][:,i][random_offset:random_offset+int((3*n)/4)]
    
    x_train[i] = better_stft(wave_).reshape((3968,))

for i in np.arange(len(x_test)):
    random_offset = np.random.randint(0, high = (int(n/4)-1) ) #-1 is for security to prevent calling out of bounds
    #random_offset = (int(n/8)-1)
    
    wave_ = mat["outData"][:,i+len(x_train)][random_offset:random_offset+int((3*n)/4)]
    
    x_test[i] = better_stft(wave_).reshape((3968,))

# ----------------- MODELLING -----------------

img_shape_ = [img_shape[0], img_shape[1], 1]
print(img_size)
inputs = keras.Input(shape=(img_size))
reshape1 = layers.Reshape((img_shape_))(inputs)
conv1 = layers.Conv2D(32, kernel_size = (5,5), strides = (1,1), activation = 'relu', input_shape=((img_shape_)))(reshape1)
max1 = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv1)
conv2 = layers.Conv2D(64, (5,5), activation = 'relu')(max1)
max2 = layers.MaxPooling2D(pool_size=(2,2))(conv2)
flat = layers.Flatten()(max2)
den1 = layers.Dense(100, activation = 'relu')(flat)
print(img_size)
decoded = layers.Dense(img_size, activation='sigmoid')(den1)
# This model maps an input to its reconstruction
autoencoder = keras.Model(inputs, decoded)

autoencoder.summary()

autoencoder.compile(optimizer = "adam", loss = "binary_crossentropy")

# ----------------- NORMALISATION -----------------

normaliserm = 0.004

normaliserc = 0
x_train = (x_train-normaliserc) / normaliserm
x_test = (x_test-normaliserc) / normaliserm

# ----------------- TRAINING -----------------

history = autoencoder.fit(x_train, x_train,
                epochs = 5, #50
                batch_size = 256, #256
                shuffle = True,
                validation_data = (x_test, x_test))

# ----------------- OUTPUTS -----------------

# Saving model
saved_model.save(autoencoder, "output/model.txt")

# Saving loss
with open("output/loss.txt", "w") as file:
    file.write("train\n")
    file.write(str(history.history["loss"]))

    file.write("\n")
    file.write("\n")

    file.write("validation\n")
    file.write(str(history.history["val_loss"]))

    file.close()