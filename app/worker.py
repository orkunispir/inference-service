from database import mariadb_connect, minio_connect
from threading import Thread

class Worker():
    def __init__(self, task_queue):
        self.mariadb_connection = mariadb_connect()
        self.minio_connection = minio_connect()
        print("Connected to mariadb and minio inside worker")
        self.task_queue = task_queue

    def start(self) -> Thread:
        worker_thread = Thread(target=self._start)
        worker_thread.start()
        print("Thread created")
        return worker_thread

    def _start(self):

        while True:
            task = self.task_queue.get(True)  # waits until a result is there, with no timeout
            
            uuid = task.uuid
            path_to_model = task.path_to_model

            print(f"Got task from queue with uuid:{uuid} and path_to_model:{path_to_model}", flush=True)

            mariadb_cursor = self.mariadb_connection.cursor()

            try:
                # write to MariaDB -> state RUNNING
                mariadb_cursor.execute("UPDATE requests SET status = 'RUNNING' WHERE uuid = %s", (uuid,))
                self.mariadb_connection.commit()
                
                print(f"Set task in MariadDB with uuid:{uuid} to \"RUNNING\"", flush=True)

                # load data from MinIO
                try:
                    # Assuming path_to_model is "bucket_name/object_name"
                    bucket_name, object_name = path_to_model.split("/", 1)
                    local_model_path = f'/tmp/{object_name}'
                    self.minio_connection.fget_object(bucket_name, object_name, local_model_path)
                except Exception as e:
                    raise Exception(f"Error loading model from MinIO: {e}")

                # execute: Load the model and train
                try:
                    # Placeholder for model loading and training logic
                    # ... your code to load the model from local_model_path and train it on the GPU
                    # ...
                    # ... (after training, assuming you have your trained model weights in 'model_weights_path')
                    model_weights_path = local_model_path
                except Exception as e:
                    raise Exception(f"Error during model execution: {e}")

                # write to MariaDB -> state DONE
                mariadb_cursor.execute("UPDATE requests SET status = 'DONE' WHERE uuid = %s", (uuid,))
                self.mariadb_connection.commit()
                
                print(f"Set task in MariadDB with uuid:{uuid} to \"DONE\"", flush=True)

                # store weights in MinIO
                try:
                    bucket_name = "trained-models"  # Choose an appropriate bucket name
                    object_name = f"model-{uuid}.pth"  # Use UUID as the object name and add model prefix
                    self.minio_connection.fput_object(bucket_name, object_name, model_weights_path)
                except Exception as e:
                    raise Exception(f"Error storing weights in MinIO: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")
                mariadb_cursor.execute("UPDATE requests SET status = 'FAILED' WHERE uuid = %s", (uuid,))
                self.mariadb_connection.commit()
            finally:
                mariadb_cursor.close()