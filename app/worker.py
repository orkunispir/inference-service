from database import mariadb_connect, minio_connect
from threading import Thread

class Worker():
    def __init__(self, task_queue):
        self.mariadb_connection = mariadb_connect()
        self.minio_connection = minio_connect()
        self.task_queue = task_queue
            
    def start(self) -> Thread:
        worker_thread = Thread(target=self._start)
        worker_thread.start()
        return worker_thread
    
    def _start(self):
        
        while True:
            uuid, path_to_model = self.task_queue.get(True) # waits until a result is there, with no timeout
            
            #print(f"received task with uuid: {uuid[1]} and path_to_data: {path_to_data[1]}")
            
            mariadb_cursor = self.mariadb_connection.cursor()
            
            # write to MariaDB -> state RUNNING
            mariadb_cursor.execute("UPDATE requests SET status = 'RUNNING' WHERE uuid = %s", (uuid))
            self.mariadb_connection.commit()

            # load data from MinIO
            try:
                # Assuming path_to_model is "bucket_name/object_name"
                bucket_name, object_name = path_to_model.split("/", 1) 
                self.minio_connection.fget_object(bucket_name, object_name, '/tmp/' + object_name) # Use a temporary file path in the container
                local_model_path = '/tmp/' + object_name # Store weights also to a temp directory
            
            except Exception as e:
                print(f"Error loading model from MinIO: {e}")
                mariadb_cursor.execute("UPDATE requests SET status = 'ERROR', message = %s WHERE uuid = %s", (str(e), uuid))
                self.mariadb_connection.commit()
                continue # Skip to the next task

            # execute: Load the model and train
            try:
                # Placeholder for model loading and training logic
                # ... your code to load the model from local_model_path and train it on the GPU
                # ...
                # ... (after training, assuming you have your trained model weights in 'model_weights_path')
                model_weights_path = local_model_path
            except Exception as e:
                print(f"Error during model execution: {e}")
                mariadb_cursor.execute("UPDATE requests SET status = 'ERROR', message = %s WHERE uuid = %s", (str(e), uuid))
                self.mariadb_connection.commit()
                continue # Skip to the next task
            
            # write to MariaDB -> state DONE
            mariadb_cursor.execute("UPDATE requests SET status = 'DONE' WHERE uuid = %s", (uuid))
            self.mariadb_connection.commit()
            
            # store weights in MinIO
            try:
                bucket_name = "trained-models"  # Choose an appropriate bucket name
                object_name = f"{uuid}.pth"  # Use UUID as the object name
                self.minio_connection.fput_object(bucket_name, object_name, model_weights_path)
            except Exception as e:
                print(f"Error storing weights in MinIO: {e}")
                mariadb_cursor.execute("UPDATE requests SET status = 'ERROR', message = %s WHERE uuid = %s", (str(e), uuid))
                self.mariadb_connection.commit()