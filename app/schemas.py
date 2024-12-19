from pydantic import BaseModel
from typing import Optional

# Message from client
class ClientRequest(BaseModel):
    bucket: str
    object_name: str
    
#Message from fastapi to multiprocess
class TrainingRequest(BaseModel):
    uuid: str
    bucket: str
    object_name: str
    
    
#Message from fastapi to MariaDB
class DatabaseRequest(BaseModel):
    uuid: str
    status: str
    bucket: str
    object_name: str
    
#Message from fastapi to client for training post
class TrainingResponse(BaseModel):
    id: str
    
#Message from fastapi to client for status get
class StatusResponse(BaseModel):
    status: str
    model_bucket: Optional[str] = None
    model_object_name: Optional[str] = None