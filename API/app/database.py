
import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker,declarative_base
from sqlalchemy.sql import func
from dotenv import load_dotenv
from sqlalchemy import Boolean
from passlib.context import CryptContext
import pathlib
#from app.logging_config import setup_logger
import yaml

current_path = pathlib.Path(__file__).parent

config_path = current_path.parent.parent / 'cluster_conf.yaml'

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

DATABASE_URL = config.get("DATABASE_URL", "sqlite:///./test.db")  # Provide a fallback if the env variable is missing
DB_DIR = config.get("DB_DIR","CURRENT_DIR")
if DB_DIR == "CURRENT_DIR":
    DB_DIR = os.getcwd()
    
db_name = config.get("DB_name","chat_bot_db")
db_path = os.path.join(DB_DIR, f"{db_name}.db")
DATABASE_URL = f"sqlite:///{db_path}"

# Check if the database file exists
if not os.path.exists(db_path):
    Base = declarative_base()
    engine = create_engine(DATABASE_URL)

DATABASE_URL = f"sqlite:///{db_path}"
# SQLAlchemy base class
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    prompt_token_number = Column(Integer, default=0)
    gen_token_number = Column(Integer, default=0)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)
    token_limit = Column(Integer, default=1000)
    role = Column(String, default="User") 
    collection_names = Column(String, default="") 

# Conversation model
class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    conversation_number = Column(Integer)  # Add conversation number column
    content = Column(String)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    conversation_name = Column(String)

# Create database engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)
with SessionLocal() as db:
    admin_username =  config.get("admin_username","admin")
    admin_password = config.get("admin_password","admin")

    existing_admin = db.query(User).filter(User.username == admin_username).first()
    if not existing_admin:
        hashed_password = pwd_context.hash(admin_password)
        new_admin = User(username=admin_username, hashed_password=hashed_password, role="Admin")
        db.add(new_admin)
        db.commit()
        print(f"Admin {admin_username} created.")
    else:
        print(f"Admin {admin_username} already exists.")
# Create tables if they don't exist


# Database session generator
def get_db():
    """
    Returns a database session.

    This function creates a new instance of the database session class `SessionLocal`
    and yields it. The session is closed once the function is done executing.

    Returns:
        SessionLocal: An instance of the `SessionLocal` database session class.
    """
    db = SessionLocal()  # SessionLocal should be your database session class
    try:
        yield db
    finally:
        db.close()
