import os
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from keras.callbacks import EarlyStopping, ModelCheckpoint
from experimental.st3dnet.ST3DNet import *
from experimental.st3dnet.metrics import *

def get_free_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    free_gpu = None
    for i, device in enumerate(physical_devices):
        try:
            # Obtém o uso da memória da GPU
            gpu_memory_info = tf.config.experimental.get_memory_info(device.name)
            gpu_memory_free = gpu_memory_info['total'] - gpu_memory_info['current']
            
            if gpu_memory_free > 0:  # Se houver memória livre, considera essa GPU
                free_gpu = device
                break
        except Exception as e:
            print(f"Erro ao verificar GPU {i}: {e}")
            
    if free_gpu is None:
        raise RuntimeError("Nenhuma GPU livre encontrada")
        
    return free_gpu

def setup_gpu():
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        gpu_id = get_free_gpu()
        tf.config.experimental.set_memory_growth(gpu_id, True)
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    return session

def build_model(c_conf, t_conf, external_dim, nb_residual_unit, lr):
    model = ST3DNet(c_conf=c_conf, t_conf=t_conf, external_dim=external_dim, nb_residual_unit=nb_residual_unit)
    adam = Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[rmse, mae, ALS])
    return model

def train_model(model, X_train, Y_train, batch_size, nb_epoch, fname_param, early_stopping, model_checkpoint):
    model.fit(X_train, Y_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)
    return model.save_weights(fname_param+'.weights.h5', overwrite=True)
     

def evaluate_model(model, X, Y, batch_size, description, file):
    score = model.evaluate(X, Y, batch_size=batch_size, verbose=0)
    file.write(f'{description} score: {score[0]:.6f}\n RMSE: {score[1]:.6f}\n MAE: {score[2]:.6f}\n ALS: {score[3]:.6f}\n')

def main(X_train, Y_train, X_test, external_dim, grid,len_closeness,len_trend):

    # session = setup_gpu()
    
    nb_epoch = 1500
    batch_size = 32
    lr = 0.0002
    nb_residual_unit = 4
    nb_flow = 2
    map_height, map_width = grid["ycell"].max() + 1, grid["xcell"].max() + 1
    # map_height, map_width = 24, 16
    
    c_conf = (len_closeness, nb_flow, map_height, map_width) if len_closeness > 0 else None
    t_conf = (len_trend, nb_flow, map_height, map_width) if len_trend > 0 else None
    
    model = build_model(c_conf, t_conf, external_dim, nb_residual_unit, lr)
    directory = os.path.join('weighted_save')
    fname_param = os.path.join(directory, "crime.best.keras")
    os.makedirs(directory, exist_ok=True)

    early_stopping = EarlyStopping(monitor='val_rmse', patience=50, mode='min')
    model_checkpoint = ModelCheckpoint(fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')
    train_model(model, X_train, Y_train, batch_size, nb_epoch, fname_param, early_stopping, model_checkpoint)
    result = model.predict(X_test)
    
    return result, model
    

