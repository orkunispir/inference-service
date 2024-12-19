import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import json

class ActivityModel:
    def __init__(self, sequence_length=50, num_features=5):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.num_classes = None  # Will be determined during preprocessing
        self.model = None  # Initialize model later after knowing num_classes

    def build_model(self):
        if self.num_classes is None:
            raise ValueError("Number of classes (num_classes) is not set. Call preprocess_data first.")

        model = Sequential()

        # First LSTM layer with Dropout and Batch Normalization
        model.add(LSTM(units=256, return_sequences=True, input_shape=(self.sequence_length, self.num_features)))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

        # Second LSTM layer with Dropout and Batch Normalization
        model.add(LSTM(units=256, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

        # Third LSTM layer with Dropout and Batch Normalization
        model.add(LSTM(units=128, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

        # Fourth LSTM layer without return sequences
        model.add(LSTM(units=128))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

        # Fully connected layer
        model.add(Dense(units=128, activation='relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())

        # Another fully connected layer to add more complexity
        model.add(Dense(units=64, activation='relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())

        # Output layer for classification (with softmax for multi-class classification)
        model.add(Dense(units=self.num_classes, activation='softmax'))

        # Compile the model for classification
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        # save model to 
        self.model = model

        return model

    def preprocess_data(self, df):
        # Encode 'Activity' as categorical
        le = LabelEncoder()
        df['Activity_Encoded'] = le.fit_transform(df['Activity'])
        # Save the label encoder for future use
        joblib.dump(le, 'label_encoder_activity.pkl')

        # Determine the number of unique classes from the data
        self.num_classes = len(le.classes_)

        df['start'] = pd.to_datetime(df['start'])
        df['end'] = pd.to_datetime(df['end'])

        df['start_hour_of_day'] = df['start'].dt.hour
        df['end_hour_of_day'] = df['end'].dt.hour
        df['start_minute_of_hour'] = df['start'].dt.minute
        df['end_minute_of_hour'] = df['end'].dt.minute

        # Drop the original 'Activity' column
        df = df.drop('Activity', axis=1)

        # Add the 'next_activity' column (shifted 'Activity_Encoded')
        df['next_activity'] = df['Activity_Encoded'].shift(-1)

        # Drop any rows with missing target values
        df = df.dropna()

        # Define feature columns and target
        feature_columns = ['start_hour_of_day', 'end_hour_of_day',
                           'start_minute_of_hour', 'end_minute_of_hour', 'Activity_Encoded']
        target_column = 'next_activity'

        # Scale the continuous time features only (exclude 'Activity_Encoded')
        scaler = MinMaxScaler()
        scaled_time_features = scaler.fit_transform(df[feature_columns[:-1]])
        # Save the scaler for future use
        joblib.dump(scaler, 'feature_scaler_activity.pkl')
        scaled_time_df = pd.DataFrame(scaled_time_features, columns=feature_columns[:-1])

        # Combine the scaled time features with 'Activity_Encoded'
        scaled_time_df['Activity_Encoded'] = df['Activity_Encoded'].values
        scaled_time_df[target_column] = df[target_column].values

        # Create sequences
        X, y = self.create_sequences(scaled_time_df, feature_columns, target_column)

        # One-hot encode y
        y_categorical = to_categorical(y, num_classes=self.num_classes)

        return X, y_categorical

    def create_sequences(self, df, feature_columns, target_column):
        X = []
        y = []
        for i in range(len(df) - self.sequence_length):
            X.append(df.iloc[i:i + self.sequence_length][feature_columns].values)
            y.append(df.iloc[i + self.sequence_length][target_column])
        return np.array(X), np.array(y)

    def train(self, X_train, y_train, epochs=10, batch_size=128, validation_data=None):
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint('best_activity_model.keras', monitor='val_loss', save_best_only=True)
        callbacks = [early_stop, checkpoint] if validation_data else [checkpoint]

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        report = classification_report(y_true, y_pred_classes, output_dict=True)  # Get report as a dictionary
        cm = confusion_matrix(y_true, y_pred_classes).tolist()  # Convert to list of lists

        print("\nClassification Report:")
        print(json.dumps(report, indent=4))  # Print report in a nicely formatted way

        print("\nConfusion Matrix:")
        print(json.dumps(cm, indent=4))  # Print confusion matrix in a nicely formatted way

        results = {
            "loss": loss,
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm,
        }

        return results

    def save_model(self, filepath):
        self.model.save(filepath)

    #def load_model(self, filepath):
    #    self.model = load_model(filepath)
    
    def split_data(self, X, y_categorical, split_ratio=0.8):
        """
        Splits the data into training and testing sets while maintaining temporal order.

        Args:
            X: Input features (sequences).
            y_categorical: One-hot encoded target variable.
            split_ratio: The proportion of the dataset to include in the train split (default: 0.8).

        Returns:
            A tuple containing: X_train, y_train, X_test, y_test.
        """
        if not isinstance(X, np.ndarray) or not isinstance(y_categorical, np.ndarray):
            raise TypeError("Input data X and y_categorical must be NumPy arrays.")

        if X.shape[0] != y_categorical.shape[0]:
            raise ValueError("Input features X and target y_categorical must have the same number of samples.")
        
        train_size = int(len(X) * split_ratio)
        X_train = X[:train_size]
        y_train = y_categorical[:train_size]
        X_test = X[train_size:]
        y_test = y_categorical[train_size:]
        return X_train, y_train, X_test, y_test