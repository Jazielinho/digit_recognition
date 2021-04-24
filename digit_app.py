'''
author: jazielinho
'''

import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from sklearn.model_selection import train_test_split
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import dill

import config


st.set_option('deprecation.showPyplotGlobalUse', False)




# =================================================================================

model = None
emb_model = None
X_train = None
y_train = None
nearest_model = None
explainer = None
segmenter = None
umap_model = None
umap_train_df = None


# =================================================================================

# PREPARANDO IMAGEN

def to_rgb(x):
    ''' Convertimos una imagen de escala de grises a RGB'''
    x_rgb = np.zeros((x.shape[0], 28, 28, 3))
    for i in range(3):
        x_rgb[..., i] = x[..., 0]
    return x_rgb.reshape(-1, 28, 28, 3)


def prepara_img(image_array):
    ''' preparamos una imagen para predecir el dígito '''
    img = cv2.resize(image_array, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(-1, 28, 28, 1)
    return to_rgb(img)


# =================================================================================
# CARGANDO Y PREDICIENDO LOS DIGITOS

def load_model():
    ''' Cargando el modelo RED NEURONAL '''
    global model
    global emb_model
    if model is None or emb_model is None:
        model_file = open(config.MODEL_PATH_JSON, 'r')
        model = model_file.read()
        model_file.close()
        model = tf.keras.models.model_from_json(model)
        model.load_weights(config.MODEL_PATH_H5)

        emb_model = tf.keras.models.Model(model.input,
                                          model.get_layer('embedding').output)
    return model, emb_model


def predict_class(img):
    ''' Calculamos y graficamos las predicciones '''
    global model
    global emb_model
    model, emb_model = load_model()
    predictions = model.predict(img)
    predictions = predictions.ravel()
    clase_predicha = int(predictions.argmax())
    prob_ = 100 * predictions[clase_predicha]
    pred_df = pd.DataFrame(predictions, index=range(10))
    st.subheader(f'Clase predicha: {clase_predicha}, probabilidad: {prob_:.2f}')
    st.bar_chart(pred_df)


# =================================================================================
# DECISIONES USANDO LIME

def load_explainer():
    ''' Cargando el modelo LIME '''
    global explainer
    global segmenter
    if explainer is None or segmenter is None:
        explainer = dill.load(open(config.MODEL_EXPLAINER, 'rb'))
        segmenter = dill.load(open(config.MODEL_SEGMENTER, 'rb'))
    return explainer, segmenter


def plot_rules(img):
    ''' Obtenemos las reglas de decisión usando LIME '''
    global model
    global emb_model
    global explainer
    global segmenter
    model, emb_model = load_model()
    explainer, segmenter = load_explainer()

    X_eval = img.reshape(28, 28, 3)

    explanation = explainer.explain_instance(X_eval,
                                             classifier_fn=model.predict,
                                             top_labels=10,
                                             hide_color=0,
                                             num_samples=100,
                                             segmentation_fn=segmenter)

    plt.figure(figsize=(15, 10))

    for i in range(10):
        temp, mask = explanation.get_image_and_mask(i,
                                                    positive_only=True,
                                                    num_features=1000,
                                                    hide_rest=False,
                                                    min_weight=0.01)
        plt.subplot(2, 5, (i + 1))
        plt.imshow(label2rgb(mask.astype(np.uint8),
                             X_eval.astype(np.uint8),
                             bg_label=0),
                   interpolation='nearest')
        plt.title(f'Positivo para clase: {i}')
        plt.axis('off')
    plt.axis('off')

    st.pyplot()

    clase_predicha = model.predict(X_eval.reshape((1, 28, 28, 3))).argmax(axis=1)[0]
    image, mask = explanation.get_image_and_mask(clase_predicha,
                                                 positive_only=True,
                                                 hide_rest=False)
    plt.imshow(X_eval.astype(np.uint8))
    plt.imshow(mark_boundaries(image.astype(np.uint8), mask))
    plt.title(f'Decisiones para la clase predicha: {clase_predicha}')
    plt.axis('off')
    st.pyplot()


# =================================================================================
# REDUCCIÓN DE DIMENSIONES USANDO UMAP

def load_umap():
    '''Cargando el modelo UMAP '''
    global umap_model
    global umap_train_df
    if umap_model is None or umap_train_df is None:
        umap_model = pickle.load(open(config.MODEL_UMAP, 'rb'))
        umap_train_df = pd.read_csv(config.UMAP_TRAIN)
    return umap_model, umap_train_df


def plot_umap(img):
    ''' Reducción de dimensiones usando UMAP '''
    global model
    global emb_model
    global umap_model
    global umap_train_df
    model, emb_model = load_model()
    umap_model, umap_train_df = load_umap()
    emb_test = emb_model.predict(img.reshape(-1, 28, 28, 3), verbose=0)
    emb_test = emb_test.reshape(1, -1)
    umap_test = umap_model.transform(emb_test)
    umap_test = umap_test.reshape(1, -1)

    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='x0', y='x1', alpha=0.1, hue='target',
                    legend='full', data=umap_train_df,
                    palette='Paired_r')
    plt.scatter(umap_test[0, 0], umap_test[0, 1], s=100, c='k')
    st.pyplot()


# =================================================================================
# IMAGENES SIMILARES USANDO NearestNeighbors

def load_data():
    ''' Cargando los datos '''
    global X_train
    global y_train
    if X_train is None or y_train is None:
        (X, y), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X = X.reshape(-1, 28, 28, 1)
        X_train, X_val, y_train, y_test = train_test_split(X, y,
                                                           test_size=0.2,
                                                           random_state=123456)
        X_train = to_rgb(X_train)
        X_train = X_train.reshape(-1, 28, 28, 3)
    return X_train, y_train


def load_nearest_model():
    ''' Cargando modelo de vecinos cercanos '''
    global nearest_model
    if nearest_model is None:
        nearest_model = pickle.load(open(config.MODEL_SKLEARN_NN, 'rb'))
    return nearest_model


def plot_similares(img):
    ''' Imágenes similares usando NearestNeighbors '''
    global model
    global emb_model
    global X_train
    global y_train
    global nearest_model
    model, emb_model = load_model()
    X_train, y_train = load_data()
    nearest_model = load_nearest_model()

    emb_test = emb_model.predict(img.reshape(1, 28, 28, 3), verbose=0)
    emb_test = emb_test.reshape(1, -1)

    distances_test, pred_nearest_test = nearest_model.kneighbors(emb_test)

    pred_nearest_test = pred_nearest_test.ravel()

    plt.figure(figsize=(15, 10))

    for i in range(nearest_model.n_neighbors):
        plt.subplot(1, nearest_model.n_neighbors, (i + 1))
        plt.imshow(X_train[pred_nearest_test[i]].astype(np.uint8),
                   cmap=plt.cm.binary)
        plt.title(y_train[pred_nearest_test[i]])
        plt.axis('off')
    plt.axis('off')
    st.pyplot()


# =================================================================================

st.title('¡RECONOCE DÍGITOS!')

st.markdown('''
La siguiente aplicación intenta predecir el dígito escrito.
* Usamos redes neuronales convolucionales con Tensorflow.
* Para identificar reglas de decisión usamos LIME.
* Para reducir dimensiones usamos UMAP.
* Para buscar imágenes similares usamos NearestNeighbors.
''')

st.markdown('''¡ESCRIBA UN DÍGITO, INTENTARÉ PREDECIRLO!''')

canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=config.SIZE_DRAW,
    height=config.SIZE_DRAW,
    drawing_mode='freedraw',
    key='canvas'
)


if canvas_result.image_data is not None:
    if st.button('PREDECIR'):
        image_array = canvas_result.image_data.astype(np.uint8)
        img = prepara_img(image_array=image_array)

        st.subheader('PREDICCIÓN')
        predict_class(img)

        st.subheader('LIME PARA REGLAS DE DECISIÓN')
        plot_rules(img)

        st.subheader('UMAP PARA REDUCCIÓN DE DIMENSIONES')
        plot_umap(img)

        st.subheader('VECINOS CERCANOS PARA BUSCAR IMÁGENES SIMILARES EN EL CONJUNTO DE ENTRENAMIENTO')
        plot_similares(img)