import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func
from dotenv import load_dotenv
from sqlalchemy import Boolean
from passlib.context import CryptContext
import yaml
import pathlib
# from app.logging_config import setup_logger

current_path = pathlib.Path(__file__).parent
config_path = current_path/ 'cluster_conf.yaml'
# Environment and DB setup
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)



DATABASE_URL = config.get("DATABASE_URL")

DB_SERVICE_URL = config.get("DB_SERVICE_URL")  # Make sure this is used somewhere in your application
DB_DIR = config.get("DB_DIR")
if DB_DIR == "CURRENT_DIR":
    DB_DIR = os.getcwd()

DB_DIR = os.path.join(DB_DIR, "API")
db_name = config.get("DB_name","chat_bot_db")
db_path = os.path.join(DB_DIR, f"{db_name}.db")
DATABASE_URL = f"sqlite:///{db_path}"

# Check if the database file exists
if not os.path.exists(db_path):
    raise FileNotFoundError("Database file not found in ",db_path)
else:
    print(f"Database file {db_path} found.")


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


# Create tables if they don't exist
# Database session generator
class Database:
    def __init__(self):
        self.db = SessionLocal()

    def add_conversation(self, input: dict):
        # Check if the user exists by username
        user = self.db.query(User).filter(User.username == input["username"]).first()
        # Retrieve the user object with the new ID

        # Calculate the conversation number for the user
        conversation_number = (
            self.db.query(Conversation).filter(Conversation.user_id == user.id).count()
            + 1
        )

        # Add the conversation
        conversation = Conversation(
            user_id=user.id,
            conversation_number=conversation_number,
            content=input["content"],
            conversation_name=input["conversation_name"],  # Save the name of the conversation
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.close()
        return {"message": "Conversation added"}

    def update_conversation(self, input):
        user = self.db.query(User).filter(User.username == input["username"]).first()

        if not user:
            self.db.close()
            return {f"User {input['username']} not found"}

        # Check if conversation_number is provided in the input
        if "conversation_number" not in input or not input["conversation_number"]:
            # Get the conversation with the highest number for the user
            conversation = (
                self.db.query(Conversation)
                .filter(Conversation.user_id == user.id)
                .order_by(Conversation.conversation_number.desc())
                .first()
            )
            # If there's no conversation for the user, create a new one
            if not conversation:
                # Calculate the conversation number for the user
                conversation_number = (
                    self.db.query(Conversation).filter(Conversation.user_id == user.id).count()
                    + 1
                )
                # Add the new conversation
                conversation = Conversation(
                    user_id=user.id,
                    conversation_number=conversation_number,
                    content=input["content"],
                    conversation_name=input.get("conversation_name", ""),  # Use the provided name or an empty string
                )
                self.db.add(conversation)
                self.db.commit()
                self.db.close()
                return {"message": "New conversation added"}
            # Otherwise, continue to update the existing conversation
        else:
            # Fetch the specified conversation using the provided number
            conversation = (
                self.db.query(Conversation)
                .filter(
                    Conversation.conversation_number == input["conversation_number"],
                    Conversation.user_id == user.id,
                )
                .first()
            )

            if not conversation:
                self.db.close()
                return {f"Conversation not found"}

        # Update conversation content
        conversation.content = input["content"]

        # Increment token numbers and check limits
        if input.get("prompt_token_number"):
            user.prompt_token_number += input["prompt_token_number"]

        if input.get("gen_token_number"):
            user.gen_token_number += input["gen_token_number"]

            # Disable user if gen_token_number exceeds token_limit
            if user.gen_token_number > user.token_limit:
                user.disabled = True
                user.token_limit = 0

        self.db.commit()
        self.db.close()
        return {"message": "Conversation updated"}

    def retrieve_conversation(self, input):
        user = self.db.query(User).filter(User.username == input["username"]).first()
        if not user:
            self.db.close()
            return {"error": f"User {input['username']} not found"}

        # Check if conversation_number is provided in the input
        if "conversation_number" in input and input["conversation_number"]:
            conversation = (
                self.db.query(Conversation)
                .filter(
                    Conversation.conversation_number == input["conversation_number"],
                    Conversation.user_id == user.id,
                )
                .first()
            )
        else:
            # Get the latest conversation for the user
            conversation = (
                self.db.query(Conversation)
                .filter(Conversation.user_id == user.id)
                .order_by(Conversation.timestamp.desc())
                .first()
            )

        if not conversation:
            self.db.close()
            return {"error": "Conversation not found"}

        self.db.close()
        return {
            "user_id": conversation.user_id,
            "conversation_number": conversation.conversation_number,
            "content": conversation.content,
            "timestamp": conversation.timestamp,
            "conversation_name": conversation.conversation_name,
        }
    def add_collection(self, input):
        user = self.db.query(User).filter(User.username == input["username"]).first()
        # the fist letter of collection name is set to uppecase because when bulding collections in Weaviate , the Weaviate will set the first letter to uppercase by default
        if not user:
            return {"error": "User not found"}
        if input["username"] and input["username"][0].isalpha():
            input["username"]= input["username"][0].upper() + input["username"][1:]

        username = input["username"]
        new_collection_name = f"{username}_{input['collection_name']}"
        if new_collection_name in user.collection_names.split(','):
            return {"collection_name": new_collection_name}

        user.collection_names += f",{new_collection_name}" if user.collection_names else new_collection_name
        self.db.commit()
        return True
    
    def check_collection_exists(self, input):
        user = self.db.query(User).filter(User.username == input["username"]).first()
        if not user:
            return {"error": "User not found"}
        if input["username"] and input["username"][0].isalpha():
            input["username"]= input["username"][0].upper() + input["username"][1:]
        username = input["username"]
        new_collection_name = f"{username}_{input['collection_name']}"  

        if new_collection_name in user.collection_names.split(','):
            return True
        return False

    def get_collections(self, input):
        user = self.db.query(User).filter(User.username == input["username"]).first()
        if not user:
            return {"error": "User not found"}

        return {"collections": user.collection_names.split(',')}
    
    def delete_collection(self, input):
        user = self.db.query(User).filter(User.username == input["username"]).first()
        if not user:
            return {"error": "User not found"}
        if input["username"] and input["username"][0].isalpha():
            input["username"]= input["username"][0].upper() + input["username"][1:]
        username = input["username"]
        new_collection_name = f"{username}_{input['collection_name']}"  

        if new_collection_name == f"{username}_General_collection":
            return {"error": "Cannot delete the default collection"}
        collection_names = user.collection_names.split(',')
        if new_collection_name not in collection_names:
            return {"error": "Collection not found"}
        
        collection_names.remove(new_collection_name)
        user.collection_names = ','.join(collection_names)
        self.db.commit()
        return {"message": "Collection deleted"}
    
    def get_all_data(self):
        self.db = SessionLocal()
        users = self.db.query(User).all()
        conversations = self.db.query(Conversation).all()

        data = []
        for user in users:
            user_data = {
                "id": user.id,
                "username": user.username,
                "prompt_token_number": user.prompt_token_number,
                "gen_token_number": user.gen_token_number,
                "timestamp": user.timestamp,
                "disabled": user.disabled,
                "token_limit": user.token_limit,
                "role": user.role,
                "collection_names": user.collection_names.split(","),
                "conversations": []
            }

            # Add user's conversations
            for conversation in conversations:
                if conversation.user_id == user.id:
                    user_data["conversations"].append({
                        "conversation_id": conversation.id,
                        "conversation_number": conversation.conversation_number,
                        "content": conversation.content,
                        "timestamp": conversation.timestamp,
                        "conversation_name": conversation.conversation_name
                    })

            data.append(user_data)

        self.db.close()
        return data
