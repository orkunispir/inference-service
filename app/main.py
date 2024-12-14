from fastapi import FastAPI, HTTPException

from schemas import *
from database import mariadb_connect
from worker import Worker

import uuid
import multiprocessing as mp

app = FastAPI()



request_queue = mp.Queue()
worker = Worker(request_queue)



@app.post("/inference/", response_model=InferenceResponse)
async def inference(request: ClientRequest):
    # create unique identifier
    id = uuid.uuid4();
    
    
    with mariadb_connect() as conn:
        
        cursor = conn.cursor()
    
        cursor.execute(f"SELECT * FROM requests WHERE path_to_model = {request.path_to_data}")
        existing_request = cursor.fetchone()

        if existing_request:
            request_id, status, *_ = existing_request  # Unpack the tuple
            if status == "RUNNING":
                # Request is already running, potentially ask Isaac (implementation omitted)
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Request already running") 
            else:
                # Old request, you might want to delete it or update it
                cursor.execute("DELETE FROM requests WHERE id = %s", (request_id))
                conn.commit()  # Commit the deletion
        
        
        # if not, store it and add to multiprocess queue
        cursor.execute("insert into requests (uuid, status, path_to_model) values (%s, %s, %s)", (str(id), "WAITING", request.path_to_data))
        
        # new inference request + add to multiprocess queue
        req = InferenceRequest(path_to_model=request.path_to_data, uuid=str(id))
        
        request_queue.put(req)
        
        print(f"Put {req} into queue.")

    
    # return uuid in response
    return InferenceResponse(id=str(id))
    