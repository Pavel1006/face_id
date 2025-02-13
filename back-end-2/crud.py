from sqlalchemy.orm import Session
import models, schemas

def create_user(db: Session, full_name: str, image_path: str, encoding: list):
    db_user = models.User(full_name=full_name, image_path=image_path, encoding=encoding)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_all_users(db: Session):
    return db.query(models.User).all()
