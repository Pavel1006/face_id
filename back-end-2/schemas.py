from pydantic import BaseModel
from typing import List

class UserBase(BaseModel):
    full_name: str

class UserCreate(UserBase):
    image_path: str
    encoding: List[float]

class UserResponse(UserBase):
    id: int

    class Config:
        orm_mode = True
