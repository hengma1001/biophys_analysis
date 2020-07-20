import os
# import sys
import time 
import h5py
import warnings 
import numpy as np
# from keras.optimizers import RMSprop

from .vae_conv import conv_variational_autoencoder
# sys.path.append('/home/hm0/Research/molecules/molecules_git/build/lib')
# from molecules.ml.unsupervised import VAE
# from molecules.ml.unsupervised import EncoderConvolution2D
# from molecules.ml.unsupervised import DecoderConvolution2D
# from molecules.ml.unsupervised.callbacks import EmbeddingCallback

# def CVAE(input_shape, hyper_dim=3):
#     optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

#     encoder = EncoderConvolution2D(input_shape=input_shape)

#     encoder._get_final_conv_params()
#     num_conv_params = encoder.total_conv_params
#     encode_conv_shape = encoder.final_conv_shape

#     decoder = DecoderConvolution2D(output_shape=input_shape,
#                                    enc_conv_params=num_conv_params,
#                                    enc_conv_shape=encode_conv_shape)

#     cvae = VAE(input_shape=input_shape,
#                latent_dim=hyper_dim,
#                encoder=encoder,
#                decoder=decoder,
#                optimizer=optimizer)
#     return cvae


def CVAE(input_shape, latent_dim=3):
    image_size = input_shape[:-1]
    channels = input_shape[-1]
    conv_layers = 4
    feature_maps = [64, 64, 64, 32]
    filter_shapes = [(3, 3), (3, 3), (3, 3), (3, 3)]
    strides = [(1, 1), (2, 2), (2, 2), (1, 1)]
    dense_layers = 1
    dense_neurons = [128]
    dense_dropouts = [0]

    feature_maps = feature_maps[0:conv_layers] 
    filter_shapes = filter_shapes[0:conv_layers] 
    strides = strides[0:conv_layers] 
    autoencoder = conv_variational_autoencoder(
        image_size, channels, conv_layers, feature_maps,
        filter_shapes, strides, dense_layers, dense_neurons, 
        dense_dropouts, latent_dim)
    autoencoder.model.summary()
    return autoencoder


def train_cvae(
        gpu_id, cm_file,
        input_size='all',
        hyper_dim=3, 
        epochs=100, 
        batch_size=1000):
    """
    A function trains convolutional variational autoencoder on assigned GPU 

    Parameters
    ----------
    gpu_id : int
        GPU label to use in training the model 
    cm_file : string 
        Path of input contact map hdf5 file 
    input_size :  int or 'all' 
        Number of frames using for training 
    hyper_dim : int 
        Number of latent dimensions 
    epochs : int 
        Number of epochs 
    batch_size : int 
        Number of frames per batch 

    Returns
    -------
    cvae : keras model 
        Keras model of trained cvae 

    """
    # read contact map from h5 file
    cm_h5 = h5py.File(cm_file, 'r', libver='latest', swmr=True)
    cm_data_input = cm_h5[u'contact_maps'].value

    # get number of frames to use for training 
    if input_size.lower() == 'all': 
        num_cm_input = len(cm_data_input) 
    elif input_size.isdigit(): 
        num_cm_input = int(input_size) 
        assert num_cm_input > batch_size, \
            "Your input size is smaller than batch size..." 
    else: 
        warnings.warn('Unrecognized frames quantity, \
                using all the frames instead')
        num_cm_input = len(cm_data_input)

    # splitting data into train and validation
    np.random.shuffle(cm_data_input)
    train_val_split = int(0.8 * num_cm_input)
    cm_data_train, cm_data_val = \
        cm_data_input[:train_val_split], \
        cm_data_input[train_val_split:num_cm_input]
    input_shape = cm_data_train.shape
    cm_h5.close()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cvae = CVAE(input_shape[1:], hyper_dim)

#     callback = EmbeddingCallback(cm_data_train, cvae)
    start_train = time.time() 
    cvae.train(cm_data_train, validation_data=cm_data_val,
               batch_size=batch_size, epochs=epochs)
    end_train = time.time()
    return cvae, end_train - start_train


def predict_from_cvae(cvae, cm_file, input_size): 
    """
    A functions predicts contact map embeddings 

    Parameters
    ----------
    cvae : keras model 
        Model for embedding prediction 
    cm_file : string 
        Path of input contact map hdf5 file 
    input_size : int 
        number of frames using for prediction 

    Returns
    -------
    cm_embeded : numpy.array 
        embedded contact maps in latent space 
    """ 
    cm_h5 = h5py.File(cm_file, 'r', libver='latest', swmr=True)
    cm_data_input = cm_h5[u'contact_maps']

    # get number of frames to use for training 
    if input_size.lower() == 'all': 
        num_cm_input = len(cm_data_input) 
    elif input_size.isdigit(): 
        num_cm_input = int(input_size) 
    else: 
        warnings.warn('Unrecognized frames quantity, \
                using all the frames instead')
        num_cm_input = len(cm_data_input)

    # return contact map embeddings 
    start_predict = time.time() 
    cm_embeded = cvae.return_embeddings(cm_data_input[:num_cm_input]) 
    end_predict = time.time() 
    return cm_embeded, end_predict - start_predict 

def h5py_save(h5_file, h5_data, h5_label): 
    cm_file = h5py.File(h5_file, 'w')
    cm_file.create_dataset(h5_label, data=h5_data)
    cm_file.close()
