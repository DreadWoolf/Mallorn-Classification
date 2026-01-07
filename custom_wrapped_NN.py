import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense

def create_nn_classifier(input_dim, numb_classifiers = 2, random_state = None)-> KerasClassifier:
    """
    Create a feed-forward neural network classifier wrapped in SciKeras'
    KerasClassifier.

    The returned classifier follows the scikit-learn API and can therefore be
    used with utilities such as cross-validation, pipelines, and stacking
    ensembles.

    Parameters
    ----------
    input_dim : int
        Number of input features (i.e., the dimensionality of each sample).
        This should correspond to the number of columns in the feature matrix
        after preprocessing. (e.g., ``X_train.shape[1]``).

    numb_classifiers : int, optional (default=2)
        Number of output classes. For binary classification, this should be 2.
        The output layer uses a softmax activation.

    Returns
    -------
    KerasClassifier
        A SciKeras-wrapped Keras Sequential model with:
        - an input layer of size `input_dim`
        - two hidden dense layers (32 and 16 units, ReLU activation)
        - a softmax output layer with `numb_classifiers` units
        - compiled using the Adam optimizer and sparse categorical cross-entropy loss
    """
    return KerasClassifier(
        model=build_model,
        model__input_dim = input_dim,
        model__numb_classifiers = numb_classifiers,
        epochs=100,
        batch_size=20,
        # verbose 1, will show info on each backpropagation.
        verbose=0,
        random_state = random_state
    )

def build_model(input_dim, numb_classifiers):
        """build model function, so Keras Classifier builds new model each time."""
        model = Sequential([
            # tf.keras.Input(shape=(x_train.shape[1],)),
            tf.keras.Input(shape=(input_dim,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(numb_classifiers, activation='softmax')
        ])
        
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            # Just to see if the accuracy each epoch while training.
            metrics=['accuracy']
        )
        return model
