from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml
import pickle


def load_model():
    with open('steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()

clf = data["model"]

st.title("MNIST Dataset Classification Using KNN Classifier")


# Load the MNIST dataset from OpenML
mnist = fetch_openml('mnist_784')
x, y = mnist["data"], mnist["target"]

# Display a random digit from the dataset
# this is only one image that will have 784 pixel data in an array.
some_digit = x.to_numpy()[36003]

some_digit_image = np.reshape(some_digit, (28, 28))
some_digit_image = some_digit_image / 255.0


# Reshape the digit array into an image
some_digit_image = np.reshape(some_digit, (28, 28))/255.0

st.write("""### Try to identify this image!""")
# Display the image
st.image(some_digit_image, caption=f"Label: {y[36003]}", width=200)

digits = (0, 1, 2, 3, 4, 5, 6, 7)

y_train_label = st.selectbox("Select a digit", digits)

ok = st.button("Predict")

if ok:
    x_train = x[:60000]
    x_test = x[60000:]
    y_train = y[:60000]
    y_test = y[60000:]

    shuffle_index = np.random.permutation(60000)
    x_train, y_train = x_train.to_numpy()[shuffle_index], y_train.to_numpy()[
        shuffle_index]

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    y_train_2 = (y_train == y_train_label)
    y_test_2 = (y_test == y_train_label)

    clf.fit(x_train, y_train_2)

    res = clf.predict([some_digit])
    st.subheader(f"The given image is predicted to be a digit {res[0]}")
