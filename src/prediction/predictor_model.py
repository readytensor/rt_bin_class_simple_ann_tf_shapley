import warnings
from typing import Optional
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback, LambdaCallback
from tensorflow.keras.losses import BinaryCrossentropy


from logger import get_logger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
warnings.filterwarnings("ignore")


MODEL_PARAMS_FNAME = "model_params.save"
MODEL_WTS_FNAME = "model_wts.save"
HISTORY_FNAME = "history.json"
COST_THRESHOLD = float("inf")


logger = get_logger(task_name="tf_model_training")


def log_epoch(epoch, logs):
    """
    Logger for the training step.

    Logs each epoch's logs to the logger including:
        - Epoch number
        - Logs from training
    """
    logger.info(f"Epoch: {epoch}")
    logger.info(f"Logs: {logs}")


class InfCostStopCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        loss_val = logs.get("loss")
        if loss_val == COST_THRESHOLD or tf.math.is_nan(loss_val):
            # print("Cost is inf, so stopping training!!")
            self.model.stop_training = True


class Classifier:
    """A wrapper class for the ANN Binary classifier in Tensorflow.

    This class provides a consistent interface that can be used with other
    classifier models.
    """

    model_name = "simple_ANN_tensorflow_binary_classifier"

    def __init__(
        self,
        D: int,
        l1_reg: Optional[float] = 1e-3,
        l2_reg: Optional[float] = 1e-1,
        lr: Optional[float] = 1e-3,
        **kwargs,
    ):
        """Construct a new binary classifier.

        Args:
            D (int): dimensionality of input data.
            l1_reg (int, optional): L1 regularization penalty.
                Defaults to 1e-3.
            l2_reg (int, optional): L2 regularization penalty.
                Defaults to 1e-1.
            lr (int, optional): Learning rate for optimizer.
                Defaults to 1e-3.
        """
        self.D = D
        self.l1_reg = np.float(l1_reg)
        self.l2_reg = np.float(l2_reg)
        self.lr = lr
        self.model = self.build_model()
        self.model.compile(
            loss=BinaryCrossentropy(),
            optimizer=Adam(learning_rate=self.lr),
            # optimizer=SGD(learning_rate=self.lr),
            metrics=["accuracy"],
        )

    def build_model(self):

        M1 = max(2, int(self.D * 1.5))
        M2 = max(5, int(self.D * 0.33))

        reg = l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        input_ = Input(self.D)
        x = input_
        x = Dense(M1, activity_regularizer=reg, activation="relu")(x)
        x = Dense(M2, activity_regularizer=reg, activation="relu")(x)
        x = Dense(1, activity_regularizer=reg, activation='sigmoid')(x)
        output_ = x
        model = Model(input_, output_)
        # model.summary()
        return model

    def fit(
            self,
            train_inputs: pd.DataFrame,
            train_targets: pd.Series,
            batch_size=100,
            epochs=1000,
        ) -> None:
        """Fit the classifier to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The labels of the training data.
        """
        tf.random.set_seed(0)
        # use 15% validation split if at least 300 samples in training data
        if train_inputs.shape[0] < 300:
            loss_to_monitor = "loss"
            validation_split=None
        else:
            loss_to_monitor = "val_loss"
            validation_split=0.15

        early_stop_callback = EarlyStopping(
            monitor=loss_to_monitor,
            min_delta=1e-3,
            patience=30
        )
        infcost_stop_callback = InfCostStopCallback()
        logger_callback = LambdaCallback(on_epoch_end=log_epoch)

        self.model.fit(
            x=train_inputs,
            y=train_targets,
            batch_size=batch_size,
            validation_split=validation_split,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            callbacks=[
                early_stop_callback,
                infcost_stop_callback,
                logger_callback,
            ],
        )

    
    def _predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class 1 probabilities for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class 1 probabilities.
        """
        return self.model.predict(inputs, verbose=1)


    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        class1_probs = self._predict(inputs, verbose=1).reshape(-1, 1)
        predicted_labels = (class1_probs >= 0.5).astype(int)
        return predicted_labels

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class probabilities.
        """
        class1_probs = self._predict(inputs).reshape(-1, 1)
        class0_probs = 1.0 - class1_probs
        probs = np.hstack((class0_probs, class1_probs))
        return probs

    def summary(self):
        """Return model summary of the Tensorflow model"""
        self.model.summary()

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the binary classifier and return the accuracy.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The labels of the test data.
        Returns:
            float: The accuracy of the binary classifier.
        """
        if self.model is not None:
            # returns list containing loss value and metric value
            # index at 1 which contains accuracy
            return self.model.evaluate(test_inputs, test_targets, verbose=0)[1]
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the binary classifier to disk.

        Args:
            model_dir_path (str): The dir path to which to save the model.
        """
        model_params = {
            "D": self.D,
            "l1_reg": self.l1_reg,
            "l2_reg": self.l2_reg,
            "lr": self.lr,
        }
        joblib.dump(model_params, os.path.join(model_dir_path, MODEL_PARAMS_FNAME))
        self.model.save_weights(os.path.join(model_dir_path, MODEL_WTS_FNAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Classifier":
        """Load the binary classifier from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Classifier: A new instance of the loaded binary classifier.
        """
        model_params = joblib.load(os.path.join(model_dir_path, MODEL_PARAMS_FNAME))
        classifier_model = cls(**model_params)
        classifier_model.model.load_weights(
            os.path.join(model_dir_path, MODEL_WTS_FNAME)
        ).expect_partial()
        return classifier_model

    def __str__(self):
        return (
            f"Model name: {self.model_name}("
            f"D: {self.D}, "
            f"l1_reg: {self.l1_reg})"
            f"l2_reg: {self.l2_reg})"
            f"lr: {self.lr})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Classifier:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data labels.
        hyperparameters (dict): Hyperparameters for the classifier.

    Returns:
        'Classifier': The classifier model
    """
    data_based_params = get_data_based_model_hyperparams(train_inputs)
    classifier = Classifier(**data_based_params, **hyperparameters)
    classifier.fit(train_inputs=train_inputs, train_targets=train_targets)
    return classifier


def predict_with_model(
    classifier: Classifier, data: pd.DataFrame, return_probs=False
) -> np.ndarray:
    """
    Predict class probabilities for the given data.

    Args:
        classifier (Classifier): The classifier model.
        data (pd.DataFrame): The input data.
        return_probs (bool): Whether to return class probabilities or labels.
            Defaults to True.

    Returns:
        np.ndarray: The predicted classes or class probabilities.
    """
    if return_probs:
        return classifier.predict_proba(data)
    return classifier.predict(data)


def save_predictor_model(model: Classifier, model_dir_path: str) -> None:
    """
    Save the classifier model to disk.

    Args:
        model (Classifier): The classifier model to save.
        model_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    model.save(os.path.join(model_dir_path))



def load_predictor_model(model_dir_path: str) -> Classifier:
    """
    Load the classifier model from disk.

    Args:
        model_dir_path (str): Dir path to the saved model.

    Returns:
        Classifier: A new instance of the loaded classifier model.
    """
    return Classifier.load(model_dir_path)


def evaluate_predictor_model(
    model: Classifier, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the classifier model and return the accuracy.

    Args:
        model (Classifier): The classifier model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the classifier model.
    """
    return model.evaluate(x_test, y_test)


def save_training_history(history, dir_path):
    """
    Save tensorflow model training history to a JSON file
    """
    hist_df = pd.DataFrame(history.history)
    hist_json_file = os.path.join(dir_path, HISTORY_FNAME)
    with open(hist_json_file, mode="w") as file_:
        hist_df.to_json(file_)


def get_data_based_model_hyperparams(data):
    """
    Set any model hyperparameters that are data dependent.
    For example, number of neurons in input layer of a neural network
    as a function of data shape.
    """
    return {"D": data.shape[1]}
