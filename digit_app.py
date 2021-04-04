

import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from skimage.segmentation import mark_boundaries
from skimage.color import gray2rgb, rgb2gray, label2rgb
import matplotlib.pyplot as plt

from typing import Tuple

import config

model = None
emb_model = None
X_train = None
nearest_model = None
explainer = None
segmenter = None


def to_rgb(x):
    x_rgb = np.zeros((x.shape[0], 28, 28, 3))
    for i in range(3):
        x_rgb[..., i] = x[..., 0]
    return x_rgb.reshape(-1, 28, 28, 3)


def load_model() -> Tuple[tf.keras.Model, tf.keras.Model]:
    ''' Cargando el modelo '''
    global model
    global emb_model
    if model is None:
        model_file = open(config.MODEL_PATH_JSON, 'r')
        model = model_file.read()
        model_file.close()
        model = tf.keras.models.model_from_json(model)
        model.load_weights(config.MODEL_PATH_H5)

        emb_model = tf.keras.models.Model(model.input, model.get_layer('embedding').output)

    return model, emb_model


def load_nearest_model():
    global nearest_model
    global X_train
    global model
    global emb_model

    if nearest_model is None:
        model, emb_model = load_model()

        (X, y), (_, _) = tf.keras.datasets.mnist.load_data()
        X = X.reshape(-1, 28, 28, 1)

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=123456)
        X_train = to_rgb(X_train)

        emb_train = emb_model.predict(X_train)

        nearest_model = NearestNeighbors(n_neighbors=config.N_NEIGHBORS, metric='cosine')
        nearest_model.fit(emb_train)

        return X_train, nearest_model


def load_explainer():
    global explainer
    global segmenter
    if explainer is None:
        explainer = lime_image.LimeImageExplainer(verbose=False, random_state=12345)
        segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

        return explainer, segmenter


# ================================================================================================================

def predict_class(img):
    global model
    global emb_model
    model, emb_model = load_model()
    predictions = model.predict(img)
    pred_df = pd.Series(predictions.ravel(), index=range(10))
    return pred_df


def get_rules(img):
    global model
    global emb_model
    global explainer
    global segmenter
    model, emb_model = load_model()
    explainer, segmenter = load_explainer()

    X_eval = img.reshape(28, 28, 3)

    explanation = explainer.explain_instance(X_eval, classifier_fn=model.predict, top_labels=10, hide_color=0,
                                             num_samples=100, segmentation_fn=segmenter)

    fig, m_axs = plt.subplots(2, 5, figsize=(12, 6))

    for i, c_ax in enumerate(m_axs.flatten()):
        temp, mask = explanation.get_image_and_mask(i, positive_only=True, num_features=1000, hide_rest=False,
                                                    min_weight=0.01)
        c_ax.imshow(label2rgb(mask.astype(np.uint8), X_eval.astype(np.uint8), bg_label=0), interpolation='nearest')
        c_ax.set_title('Positive for {}'.format(i))
        c_ax.axis('off')

    image, mask = explanation.get_image_and_mask(model.predict(X_eval.reshape((1, 28, 28, 3))).argmax(axis=1)[0],
                                                 positive_only=True, hide_rest=False)
    fig, ax = plt.subplots()

    ax.imshow(X_eval.astype(np.uint8))
    ax.imshow(mark_boundaries(image.astype(np.uint8), mask))

    return m_axs, ax
    # return ax


def get_similars(img):
    global model
    global emb_model
    global nearest_model
    global X_train
    model, emb_model = load_model()
    emb_test = emb_model.predict(img.reshape(1, 28, 28, 3), verbose=1)
    emb_test = emb_test.reshape(1, -1)

    X_train, nearest_model = load_nearest_model()
    distances_test, pred_nearest_test = nearest_model.kneighbors(emb_test)

    pred_nearest_test = pred_nearest_test.ravel()

    plt.figure(figsize=(15, 10))

    for i in range(len(pred_nearest_test)):
        plt.subplot(1, config.N_NEIGHBORS, i)
        plt.imshow(X_train[pred_nearest_test[i]].astype(np.uint8), cmap=plt.cm.binary)

    plt.axis('off')

    return plt.show()





# ================================================================================================================
st.title('My Digit Recognizer')
st.markdown(''' Try to write a digit! ''')

mode = st.checkbox("Draw (or Delete)?", True)

canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=config.SIZE_DRAW,
    height=config.SIZE_DRAW,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')


if canvas_result.image_data is not None:
    if st.button('Predict'):
        img = cv2.resize(canvas_result.image_data.astype(np.uint8), (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(-1, 28, 28, 1)
        img = to_rgb(img)

        pred_df = predict_class(img)
        st.bar_chart(pred_df)

        similares = get_similars(img)
        st.pyplot(similares)

        # all_rules, rules = get_rules(img)
        # st.pyplot(all_rules.figure)
        # st.pyplot(rules.figure)










