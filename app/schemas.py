from pydantic import BaseModel

# Message from client
class ClientRequest(BaseModel):
    path_to_data: str
    
#Message from fastapi to multiprocess
class InferenceRequest(BaseModel):
    uuid: str
    path_to_model: str
    
    
#Message from fastapi to MariaDB
class DatabaseRequest(BaseModel):
    uuid: str
    status: str
    path_to_model: str
    
#Message from fastapi to client
class InferenceResponse(BaseModel):
    id: str