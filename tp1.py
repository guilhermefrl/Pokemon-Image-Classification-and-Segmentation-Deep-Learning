#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aprendizagem Profunda, TP1
"""
import os as operative_system
operative_system.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from datetime import datetime
import tp1_utils
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0


# Carregar os dados
dataset = tp1_utils.load_data()

train_X = dataset["train_X"]
test_X = dataset["test_X"]
train_masks = dataset["train_masks"]
test_masks = dataset["test_masks"]
train_classes = dataset["train_classes"]
test_classes = dataset["test_classes"]
train_labels = dataset["train_labels"]
test_labels = dataset["test_labels"]

# Fazer reshape dos dados para arrays de 64x64, com 3 canais, pois são imagens a cores
train_X = train_X.reshape((train_X.shape[0], 64, 64, 3))
test_X = test_X.reshape((test_X.shape[0], 64, 64, 3))

# Dividir o conjunto de treino em conjunto de validação (500) e treino (3500)
validation_size=3500

validation_X = train_X[validation_size:]
train_X = train_X[:validation_size]

validation_masks = train_masks[validation_size:]
train_masks = train_masks[:validation_size]

validation_classes = train_classes[validation_size:]
train_classes = train_classes[:validation_size]

validation_labels = train_labels[validation_size:]
train_labels = train_labels[:validation_size]


# Main
def main():
    # 1 - Multiclass classification
    #Multiclass_Classification()

    # 2 - Multilabel classification
    Multilabel_Classification()

    # 3 - Semantic segmentation
    #Semantic_Segmentation()

    # 4 - Transfer Learning
    #Multiclass_Classification_tl()
    #Multilabel_Classification_tl()


# 1 - Multiclass classification

# Função para criar o modelo
def Create_Model_Multiclass_Classification():

    # Criar um modelo sequencial
    model = Sequential()

    # Camada convolucional 2D, com 32 filtros, kernel 3x3 e same padding
    model.add(Conv2D(32, (2, 2), padding="same", input_shape=(64,64,3)))
    # Ativação ReLU
    model.add(Activation("relu"))
    # Normalização do Batch
    model.add(BatchNormalization())
    # MaxPooling
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Camada convolucional 2D, com 64 filtros, kernel 3x3 e same padding
    model.add(Conv2D(64, (3, 3), padding="same"))
    # Ativação ReLU
    model.add(Activation("relu"))
    # Normalização do Batch
    model.add(BatchNormalization())
    # MaxPooling
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Camada convolucional 2D, com 64 filtros, kernel 3x3 e same padding
    model.add(Conv2D(64, (3, 3), padding="same"))
    # Ativação ReLU
    model.add(Activation("relu"))
    # Normalização do Batch
    model.add(BatchNormalization())
    # MaxPooling
    model.add(MaxPooling2D(pool_size=(4, 4)))

    # Camada convolucional 2D, com 64 filtros, kernel 3x3 e same padding
    model.add(Conv2D(64, (3, 3), padding="same"))
    # Ativação ReLU
    model.add(Activation("relu"))
    # Normalização do Batch
    model.add(BatchNormalization())
    # MaxPooling
    model.add(MaxPooling2D(pool_size=(4, 4)))

    # Passar input tridimensional para um vetor de uma dimensão
    model.add(Flatten())

    # Camada densa de 512 neurónios
    model.add(Dense(512))
    # Ativação ReLU
    model.add(Activation("relu"))
    # Batch Normalization
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.5))

    # Camada densa de 10 neurónios, pois são 10 classes diferentes
    model.add(Dense(10))
    # Ativação Softmax
    model.add(Activation("softmax"))

    return model


# Função para executar o modelo da Multiclass Classification
def Multiclass_Classification():

    # Taxa de Aprendizagem
    INITIAL_LEARNING_RATE = 0.005
    # Número de Epochs
    NUM_EPOCHS = 40
    # Momentum
    MOMENTUM = 0.9

    # Inicializar o otimizador Stochastic Gradient Descent
    opt = SGD(lr=INITIAL_LEARNING_RATE, momentum=MOMENTUM, decay=INITIAL_LEARNING_RATE / NUM_EPOCHS)
    # Criar o modelo
    model = Create_Model_Multiclass_Classification()
    # Compilar o modelo
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    # Mostrar o sumário do modelo
    model.summary()

    # Criar gráfico no TensorBoard
    tensorboard_callback = Tensorboard_Graph()

    # Tamanho do Batch
    BATCH_SIZE = 32
    # Treinar o modelo
    model.fit(train_X, train_classes, validation_data=(validation_X, validation_classes), 
        batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, callbacks=[tensorboard_callback])

    # Avaliar modelo
    model.evaluate(test_X, test_classes) 


# 2 - Multilabel classification

# Função para criar o modelo
def Create_Model_Multilabel_Classification():

    # Criar um modelo sequencial
    model = Sequential()

    # Camada convolucional 2D, com 32 filtros, kernel 3x3 e same padding
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(64,64,3)))
    # Ativação ReLU
    model.add(Activation("relu"))
    # Normalização do Batch
    model.add(BatchNormalization())
    # MaxPooling
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Camada convolucional 2D, com 64 filtros, kernel 3x3 e same padding
    model.add(Conv2D(64, (3, 3), padding="same"))
    # Ativação ReLU
    model.add(Activation("relu"))
    # Normalização do Batch
    model.add(BatchNormalization())
    # MaxPooling
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Camada convolucional 2D, com 64 filtros, kernel 3x3 e same padding
    model.add(Conv2D(64, (3, 3), padding="same"))
    # Ativação ReLU
    model.add(Activation("relu"))
    # Normalização do Batch
    model.add(BatchNormalization())
    # MaxPooling
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Camada convolucional 2D, com 64 filtros, kernel 3x3 e same padding
    model.add(Conv2D(64, (3, 3), padding="same"))
    # Ativação ReLU
    model.add(Activation("relu"))
    # Normalização do Batch
    model.add(BatchNormalization())
    # MaxPooling
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Passar input tridimensional para um vetor de uma dimensão
    model.add(Flatten())

    # Camada densa de 256 neurónios
    model.add(Dense(256))
    # Ativação ReLU
    model.add(Activation("relu"))
    # Batch Normalization
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.5))

    # Camada densa de 64 neurónios
    model.add(Dense(64))
    # Ativação ReLU
    model.add(Activation("relu"))
    # Batch Normalization
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.5))

    # Camada densa de 10 neurónios, pois são 10 classes diferentes
    model.add(Dense(10))
    # Ativação Sigmoid
    model.add(Activation("sigmoid"))

    return model


# Função para executar o modelo da Multilabel Classification
def Multilabel_Classification():

    # Taxa de Aprendizagem
    INITIAL_LEARNING_RATE = 0.001
    # Número de Epochs
    NUM_EPOCHS = 30

    # Inicializar o otimizador Adam
    opt = Adam(lr=INITIAL_LEARNING_RATE)
    # Criar o modelo
    model = Create_Model_Multilabel_Classification()
    # Compilar o modelo
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["binary_accuracy"])
    # Mostrar o sumário do modelo
    model.summary()

    # Criar gráfico no TensorBoard
    tensorboard_callback = Tensorboard_Graph()

    # Tamanho do Batch
    BATCH_SIZE = 20
    # Treinar o modelo
    model.fit(train_X, train_labels, validation_data=(validation_X, validation_labels), 
        batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, callbacks=[tensorboard_callback])

    # Avaliar modelo
    model.evaluate(test_X, test_labels) 


# 3 - Semantic segmentation

# Função para criar o modelo
def Create_Model_Semantic_Segmentation():
    
    # U-Net
    inputs = Input(shape=(64,64,3))
    
    # Parte do encoder
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)
    
    # Camada intermédia
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # Parte do decoder
    u5 = UpSampling2D(size=(2, 2))(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    u6 = UpSampling2D(size=(2, 2))(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.1)(c6)
    c6 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = UpSampling2D(size=(2, 2))(c6)
    u7 = concatenate([u7, c1], axis=3)
    c7 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.1)(c7)
    c7 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)
        
    model = Model(inputs=inputs, outputs=outputs)

    return model


# Função para executar o modelo da Semantic segmentation
def Semantic_Segmentation():

    # Taxa de Aprendizagem
    INITIAL_LEARNING_RATE = 0.001
    # Número de Epochs
    NUM_EPOCHS = 5

    # Inicializar o otimizador Adam
    opt = Adam(lr=INITIAL_LEARNING_RATE)
    # Criar o modelo
    model = Create_Model_Semantic_Segmentation()
    # Compilar o modelo
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    # Mostrar o sumário do modelo
    model.summary()

    # Criar gráfico no TensorBoard
    tensorboard_callback = Tensorboard_Graph()

    # Tamanho do Batch
    BATCH_SIZE = 20
    # Treinar o modelo
    model.fit(train_X, train_masks, validation_data=(validation_X, validation_masks), 
        batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, callbacks=[tensorboard_callback])

    # Avaliar modelo
    model.evaluate(test_X, test_masks) 

    # Exportar imagens
    tp1_utils.images_to_pic('test_set.png', test_X[:40], width=10)

    # Comparar as máscaras
    predicts = model.predict(test_X)
    tp1_utils.compare_masks('test_predicted.png', test_masks[:40], predicts[:40], width=10)

    # Overlay das máscaras
    tp1_utils.overlay_masks('test_overlay.png', test_X[:40], predicts[:40], width=10)


# Função para criar gráfico no TensorBoard
def Tensorboard_Graph():
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "logs"
    log_dir = "{}/model-{}/".format(root_logdir, now)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    return tensorboard_callback


# 4 - Transfer Learning

def Create_Model_Multiclass_Classification_tl():
    inputs = Input(shape=(64,64,3))

    # Carregar o modelo
    model_keras = EfficientNetB0(weights='imagenet', input_tensor=inputs , include_top=False)

    # Desativar o treino, de forma a que os parâmetros não sejam alterados
    for layer in model_keras.layers:
        layer.trainable = False

    # Criar um modelo sequencial
    model = Sequential()

    # Adicionar o modelo já treinado
    model.add(model_keras)

    # Passar input tridimensional para um vetor de uma dimensão
    model.add(Flatten())

    # Camada densa de 512 neurónios
    model.add(Dense(512))
    # Ativação ReLU
    model.add(Activation("relu"))
    # Batch Normalization
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.5))

    # Camada densa de 10 neurónios, pois são 10 classes diferentes
    model.add(Dense(10))
    # Ativação Softmax
    model.add(Activation("softmax"))

    return model


def Multiclass_Classification_tl():

    # Taxa de Aprendizagem
    INITIAL_LEARNING_RATE = 0.005
    # Número de Epochs
    NUM_EPOCHS = 40
    # Momentum
    MOMENTUM = 0.9

    # Inicializar o otimizador Stochastic Gradient Descent
    opt = SGD(lr=INITIAL_LEARNING_RATE, momentum=MOMENTUM, decay=INITIAL_LEARNING_RATE / NUM_EPOCHS)
    # Criar o modelo
    model = Create_Model_Multiclass_Classification_tl()
    # Compilar o modelo
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    # Mostrar o sumário do modelo
    model.summary()

    # Criar gráfico no TensorBoard
    tensorboard_callback = Tensorboard_Graph()

    # Tamanho do Batch
    BATCH_SIZE = 32
    # Treinar o modelo
    model.fit(train_X, train_classes, validation_data=(validation_X, validation_classes), 
        batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, callbacks=[tensorboard_callback])

    # Avaliar modelo
    model.evaluate(test_X, test_classes)


def Create_Model_Multilabel_Classification_tl():
    inputs = Input(shape=(64,64,3))

    # Carregar o modelo
    model_keras = MobileNetV2(weights='imagenet', input_tensor=inputs , include_top=False)

    # Desativar o treino, de forma a que os parâmetros não sejam alterados
    for layer in model_keras.layers:
        layer.trainable = False

    # Criar um modelo sequencial
    model = Sequential()

    # Adicionar o modelo já treinado
    model.add(model_keras)

    # Passar input tridimensional para um vetor de uma dimensão
    model.add(Flatten())

    # Camada densa de 256 neurónios
    model.add(Dense(256))
    # Ativação ReLU
    model.add(Activation("relu"))
    # Batch Normalization
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.5))

    # Camada densa de 64 neurónios
    model.add(Dense(64))
    # Ativação ReLU
    model.add(Activation("relu"))
    # Batch Normalization
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.5))

    # Camada densa de 10 neurónios, pois são 10 classes diferentes
    model.add(Dense(10))
    # Ativação Sigmoid
    model.add(Activation("sigmoid"))

    return model


def Multilabel_Classification_tl():

    # Taxa de Aprendizagem
    INITIAL_LEARNING_RATE = 0.001
    # Número de Epochs
    NUM_EPOCHS = 30

    # Inicializar o otimizador Adam
    opt = Adam(lr=INITIAL_LEARNING_RATE)
    # Criar o modelo
    model = Create_Model_Multilabel_Classification_tl()
    # Compilar o modelo
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["binary_accuracy"])
    # Mostrar o sumário do modelo
    model.summary()

    # Criar gráfico no TensorBoard
    tensorboard_callback = Tensorboard_Graph()

    # Tamanho do Batch
    BATCH_SIZE = 20
    # Treinar o modelo
    model.fit(train_X, train_labels, validation_data=(validation_X, validation_labels), 
        batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, callbacks=[tensorboard_callback])

    # Avaliar modelo
    model.evaluate(test_X, test_labels) 


if __name__ == "__main__":
    main()