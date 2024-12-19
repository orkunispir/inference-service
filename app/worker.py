from database import mariadb_connect, minio_connect
from threading import Thread
from minio import S3Error
import pandas as pd
import os
from model import ActivityModel
import json

class Worker():
    def __init__(self, task_queue):
        self.mariadb_connection = mariadb_connect()
        self.minio_connection = minio_connect()
        print("Connected to mariadb and minio inside worker")
        self.task_queue = task_queue

    def start(self) -> Thread:
        worker_thread = Thread(target=self.run)
        worker_thread.start()
        print("Thread created")
        return worker_thread

    def run(self):

        while True:
            task = self.task_queue.get(True)  # waits until a result is there, with no timeout
            
            uuid = task.uuid
            bucket_name = task.bucket
            object_name = task.object_name

            print(
                f"Got task from queue with uuid: {uuid}, bucket: {bucket_name}, object_name: {object_name}",
                flush=True,
            )

            mariadb_cursor = self.mariadb_connection.cursor()

            try:
                # write to MariaDB -> state RUNNING
                mariadb_cursor.execute(
                    "UPDATE requests SET status = 'TRAINING', updated_at = NOW() WHERE uuid = %s",
                    (uuid,),
                )
                self.mariadb_connection.commit()
                
                print(f"Set task in MariadDB with uuid:{uuid} to \"TRAINING\"", flush=True)

                # load data from MinIO
                try:
                    local_data_path = f"/tmp/{object_name}"  # Temporary path for download
                    self.minio_connection.fget_object(
                        bucket_name, object_name, local_data_path
                    )
                    print(
                        f"Successfully downloaded object '{object_name}' from bucket '{bucket_name}' to '{local_data_path}'",
                        flush=True,
                    )
                except S3Error as e:
                    raise Exception(f"Error loading object from MinIO: {e}")

                # execute: Load the model and train
                try:
                    df = pd.read_csv(local_data_path)
                    
                    sequence_length = 50  # You can adjust this
                    num_features = 5  # You can adjust this if you add more features in the future
                    model = ActivityModel(sequence_length=sequence_length, num_features=num_features)
                    
                    X, y_categorical = model.preprocess_data(df.copy())
                    
                    X_train, y_train, X_test, y_test = model.split_data(X, y_categorical, split_ratio=0.8)
                    
                    train_history = model.train(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))
                    
                    eval_results = model.evaluate(X_test, y_test)
                    
                    trained_model_path = f"/tmp/trained_model_{uuid}.keras"
                    
                    model.save_model(trained_model_path)
                    
                    print(f"Trained model saved locally to {trained_model_path}")
                    
                    # Save training history to a dictionary
                    train_history_dict = train_history.history

                    results = {
                        "train_history": train_history_dict,
                        "evaluation": eval_results,  # Now contains the dictionary from model.evaluate()
                    }

                    # Save the combined results to a JSON file
                    results_path = f"/tmp/results_{uuid}.json"
                    with open(results_path, "w") as f:
                        json.dump(results, f, indent=4)

                    print(f"Training history and evaluation results saved to {results_path}")
                    
                except Exception as e:
                    raise Exception(f"Error during model execution: {e}")
                
                # store weights in MinIO
                trained_model_bucket = "trained-models"
                trained_models_folder = str(uuid)
                trained_model_object_name = f"{trained_models_folder}/trained_model.keras"
                trained_model_results_object_name = f"{trained_models_folder}/results.json"
                
                
                
                try:
                    
                    # Check if the bucket exists
                    if not self.minio_connection.bucket_exists(trained_model_bucket):
                        # Create the bucket if it doesn't exist
                        self.minio_connection.make_bucket(trained_model_bucket)
                        print(f"Bucket '{trained_model_bucket}' created.")
                        
                    
                    self.minio_connection.fput_object(
                        trained_model_bucket,
                        trained_model_object_name,
                        trained_model_path,
                    )
                    print(
                        f"Trained model saved to MinIO: {trained_model_bucket}/{trained_model_object_name}"
                    )
                    
                    self.minio_connection.fput_object(
                        trained_model_bucket,
                        trained_model_results_object_name,
                        results_path,
                    )
                    
                    print(f"Training history and evaluation results saved to MinIO: {trained_model_bucket}/{trained_model_results_object_name}")
                
                except S3Error as e:
                    raise Exception(f"Error saving trained model to MinIO: {e}")

                # write to MariaDB -> state DONE
                mariadb_cursor.execute(
                    "UPDATE requests SET status = 'READY', model_bucket = %s, model_object_name = %s, updated_at = NOW() WHERE uuid = %s",
                    (trained_model_bucket, trained_model_object_name, uuid),
                )
                self.mariadb_connection.commit()
                
                print(f"Set task in MariadDB with uuid:{uuid} to \"DONE\"", flush=True)


            except Exception as e:
                print(f"An error occurred: {e}")
                mariadb_cursor.execute("UPDATE requests SET status = 'FAILED' WHERE uuid = %s", (uuid,))
                self.mariadb_connection.commit()
            finally:
                mariadb_cursor.close()