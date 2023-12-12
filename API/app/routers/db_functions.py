from fastapi import APIRouter, Depends, HTTPException, status, Security
from sqlalchemy.orm import Session
from app.models import LoginUser, Input, DataBaseRequest
from app.depencencies.security import get_current_active_user
from app.database import get_db
from app.database import User, Conversation
from functools import wraps
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def role_required(role: str):
    """
    A decorator function that checks if the current user has the specified role.

    Parameters:
        role (str): The role that the current user must have.

    Returns:
        decorator: The decorator function that wraps the original function.

    Raises:
        HTTPException: If the current user does not have the specified role, raises an HTTPException with status code 403 and detail "Permission Denied".
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get(
                "current_user"
            )  # current_user is provided by dependencies
            if current_user and current_user.role == role:
                return await func(*args, **kwargs)
            else:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, detail="Permission Denied"
                )

        return wrapper

    return decorator


router = APIRouter()
# ------------------------------- Admin Access Methods --------------------------------


@router.post("/add_user/")
@role_required("Admin")
async def add_user(
    input: Input,
    current_user: LoginUser = Security(
        get_current_active_user
    ),
    db: Session = Depends(get_db),
):
    """
    Adds a new user to the system.

    Parameters:
    - input: An instance of the Input class containing the user details.
    - current_user: An instance of the LoginUser class representing the currently logged-in user.
    - db: An instance of the Session class representing the database session.

    Returns:
    - A dictionary with the username of the current user.

    Raises:
    - HTTPException(400): If the username already exists.

    Exceptions:
    - Exception: If an error occurs during the execution of the function.

    Description:
    This function adds a new user to the system. It first checks if the username already exists in the database. If it does, 
    it raises an HTTPException with a status code of 400 and a detail message indicating that the user already exists. 
    If the username is unique, it hashes the user's password using the pwd_context, creates a new User instance, 
    and sets the prompt_token_number and gen_token_number attributes. If the input includes a token_limit, 
    it sets the token_limit attribute of the user accordingly. Finally, it adds the user to the database, commits the transaction, 
    refreshes the user object, and returns a dictionary with the username of the current user.

    If any exception occurs during the execution of the function, it is caught in the except block and a dictionary 
    with an error message is returned.

    Note: This function requires the user to have the "Admin" role.
    """
    try:
        # Check if the username already exists
        existing_user = db.query(User).filter(User.username == input.username).first()
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="User already exists. Please choose another username.",
            )

        hashed_password = pwd_context.hash(input.password)
        collection_username = input.username
        # the fist letter of collection name is set to uppecase because when bulding collections in Weaviate , the Weaviate will set the first letter to uppercase by default
        if collection_username and collection_username[0].isalpha():
            collection_username= collection_username[0].upper() + collection_username[1:]
        user = User(
            username=input.username,
            hashed_password=hashed_password,
            prompt_token_number=0,
            gen_token_number=0,
            collection_names= f"{collection_username}_General_collection",
        )

        # Set token_limit if provided
        if input.token_limit is not None:
            user.token_limit = input.token_limit
        else:
            user.token_limit = 100

        db.add(user)
        db.commit()
        db.refresh(user)
        return {"success": "User added with username: " + user.username}
    except Exception as e:
        # Handle the exception here
        return {"failed": "An error occurred: " + str(e)}

#--------------------------------
@router.post("/update_token_limit/")
@role_required("Admin")  # Only an Admin can add a update_token_limit
async def update_token_limit(
    input: Input,
    current_user: LoginUser = Security(get_current_active_user),
    db: Session = Depends(get_db),  #
):
    """
    Updates the token limit for a user.

    Parameters:
        - input: The input object containing the username and the new token limit. (Type: Input)
        - current_user: The currently logged-in user. (Type: LoginUser)
        - db: The database session. (Type: Session)

    Returns:
        A dictionary with a message indicating the success or failure of the token limit update. (Type: Dict[str, str])
    """
    try:
        user = db.query(User).filter(User.username == input.username).first()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        user.token_limit = input.token_limit

        db.commit()

        return {"message": "Token limit updated"}
    except Exception as e:
        # Handle the exception here
        return {"message": "An error occurred: " + str(e)}

#--------------------------------
@router.get("/get_all_data/")
@role_required("Admin")  # Only an Admin can add a user
async def get_all_data(
    current_user: LoginUser = Security(get_current_active_user),
    db: Session = Depends(get_db),  #
):
    """
    Retrieves all data from the database, including information about users and their conversations.

    Parameters:
        current_user (LoginUser): The currently logged-in user.
        db (Session): The database session.

    Returns:
        list: A list of dictionaries containing user data, including their conversations.

    Raises:
        Exception: If an error occurs while retrieving the data.
    """
    try:
        users = db.query(User).all()
        data = []

        for user in users:
            user_data = {
                "user_id": user.id,
                "username": user.username,
                "prompt_token_number": user.prompt_token_number,
                "gen_token_number": user.gen_token_number,
                "conversations": [],
                "token_limit": user.token_limit,
                "role": user.role,
                "disabled": user.disabled,
                "collection_names": user.collection_names
            }

            conversations = (
                db.query(Conversation)
                .filter(Conversation.user_id == user.id)
                .order_by(Conversation.conversation_number)  # Order by conversation number
                .all()
            )

            for conversation in conversations:
                user_data["conversations"].append(
                    {
                        "conversation_number": conversation.conversation_number,
                        "content": conversation.content,
                        "timestamp": conversation.timestamp,
                    }
                )

            data.append(user_data)

        return data
    except Exception as e:
        # Handle the exception here
        return {"message": "An error occurred: " + str(e)}

#--------------------------------------------

@router.delete("/delete_user/")
@role_required("Admin")
async def delete_user(
    input: Input,
    current_user: LoginUser = Security(
        get_current_active_user
    ),  # Use Security for dependencies that return a user
    db: Session = Depends(get_db),
):
    """
    Delete a user and all related content.

    Parameters:
    - input: The input data for deleting a user.
    - current_user: The currently logged-in user. This is a security dependency.
    - db: The database session.

    Returns:
    A dictionary with the message "User and related content deleted" if the user was successfully deleted, or a dictionary with the message "An error occurred: {error_message}" if an error occurred during the deletion process.
    """
    try:
        user = db.query(User).filter(User.username == input.username).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        db.query(Conversation).filter(Conversation.user_id == user.id).delete()
        db.delete(user)
        db.commit()

        return {"message": "User and related content deleted"}
    except Exception as e:
        # Handle the exception here
        return {"message": "An error occurred: " + str(e)}
#---------------------------------------------------

@router.post("/check_user_existence/")
@role_required("Admin")
async def check_user_existence(
    input: Input,
    current_user: LoginUser = Security(
        get_current_active_user
    ),  # Use Security for dependencies that return a user
    db: Session = Depends(get_db),
):
    """
    Check if a user exists in the database.

    Args:
        input (Input): The input data containing the username.
        current_user (LoginUser, optional): The currently logged in user. Defaults to Security(get_current_active_user).
        db (Session, optional): The database session. Defaults to Depends(get_db).

    Returns:
        Dict[str, Union[bool, str]]: A dictionary containing the result of the user existence check.
            - If the user exists, it returns {"user_exists": True}.
            - If the user does not exist, it returns {"user_exists": False}.
            - If an error occurs, it returns {"error": str(e)}.
    """
    try:
        user = db.query(User).filter(User.username == input.username).first()
        if user:
            return {"user_exists": True}  # Return the hashed password
        else:
            return {"user_exists": False}
    except Exception as e:
        return {"error": str(e)}


@router.put("/disable_user/")
@role_required("Admin")
async def disable_user(
    input: Input,
    current_user: LoginUser = Security(
        get_current_active_user
    ),  # Use Security for dependencies that return a user
    db: Session = Depends(get_db),
):
    """
    Disable a user by setting the 'disabled' flag to True in the database.
    
    Parameters:
    - input (Input): The input data containing the username of the user to be disabled.
    - current_user (LoginUser): The currently logged-in user. This is obtained from a security dependency.
    - db (Session): The database session object.
    
    Returns:
    - dict: A dictionary containing the message "User disabled" if the user was successfully disabled.
             If an error occurs, it returns a dictionary with the key "error" and the error message.
    """
    try:
        user = db.query(User).filter(User.username == input.username).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        user.disabled = True
        db.commit()
        return {"message": "User disabled"}
    except Exception as e:
        return {"error": str(e)}


@router.post("/update_token_number/")
@role_required("Admin")
async def update_token_number(
    input: Input,
    current_user: LoginUser = Security(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Update the token number for a user.

    Parameters:
        - input: The input data for updating the token number.
        - current_user: The currently logged-in user.
        - db: The database session.

    Returns:
        - A dictionary with a message indicating that the token number has been updated,
          or an error message if an exception occurs.
    """
    try:
        user = db.query(User).filter(User.username == input.username).first()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Update gen_token_number and check limits
        if input.gen_token_number is not None:
            user.gen_token_number += input.gen_token_number

            # Disable user if gen_token_number exceeds token_limit
            if user.gen_token_number > user.token_limit:
                user.disabled = True
                user.token_limit = 0

        db.commit()

        return {"message": "Gen token number updated"}
    except Exception as e:
        # Handle the exception here
        return {"error": str(e)}
# ------------------------------------ user Access methods ------------------------------------
@router.post("/add_conversation/")
async def add_conversation(
    input: Input,
    current_user: LoginUser = Security(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Adds a conversation to the database.

    Args:
        input (Input): The input data for the conversation.
        current_user (LoginUser): The currently logged-in user.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing a success message if the conversation was added successfully, or an error message if there was an exception.
    """
    try:
        # Check if the user exists by username
        user = db.query(User).filter(User.username == current_user.username).first()
        if not user:
            # If the user doesn't exist, add it using the add_user function
            user_id = add_user(input)
            user = db.query(User).get(
                user_id["user_id"]
            )  # Retrieve the user object with the new ID
        else:
            user_id = user.id

        # Calculate the conversation number for the user
        conversation_number = (
            db.query(Conversation).filter(Conversation.user_id == user.id).count() + 1
        )

        # Add the conversation
        conversation = Conversation(
            user_id=user.id,
            conversation_number=conversation_number,
            content=input.content,
            conversation_name=input.conversation_name,  # Save the name of the conversation
        )
        db.add(conversation)
        db.commit()
        return {"success": "Conversation added"}
    except Exception as e:
        return {"error": str(e)}


@router.delete("/delete_conversation/")
async def delete_conversation(
    input: Input,
    current_user: LoginUser = Security(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Deletes a conversation from the database.

    Parameters:
        - input (Input): The input data with the conversation details.
        - current_user (LoginUser): The current logged-in user.
        - db (Session): The database session.

    Returns:
        - dict: A dictionary with the response message or error details.
    """
    try:
        user = db.query(User).filter(User.username == current_user.username).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.conversation_number == input.conversation_number,
                Conversation.user_id == user.id,
            )
            .first()
        )

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        db.delete(conversation)
        db.commit()

        return {"message": "Conversation deleted"}
    except Exception as e:
        return {"error": str(e)}


@router.post("/retrieve_conversation/")
async def retrieve_conversation(
    input: Input,
    current_user: LoginUser = Security(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Retrieves a conversation from the database based on the input parameters.

    Args:
        input (Input): The input parameters for retrieving the conversation.
        current_user (LoginUser, optional): The currently logged-in user. Defaults to the active user.
        db (Session, optional): The database session. Defaults to the global session.

    Returns:
        dict: A dictionary containing the retrieved conversation details, including the user ID, conversation ID, content, timestamp, and conversation name.

    Raises:
        HTTPException: If the user or conversation is not found in the database.
    """
    try:
        user = db.query(User).filter(User.username == current_user.username).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.conversation_number == input.conversation_number,
                Conversation.user_id == user.id,
            )
            .first()
        )

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {
            "user_id": conversation.user_id,
            "conversation_id": conversation.conversation_number,
            "content": conversation.content,
            "timestamp": conversation.timestamp,
            "conversation_name": conversation.conversation_name,
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/retrieve_latest_conversation/")
async def retrieve_latest_conversation(input: DataBaseRequest,
    current_user: LoginUser = Security(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Retrieves the latest conversation for a given user.

    Parameters:
        input (Input): The input object containing the username.
        current_user (LoginUser): The currently logged-in user.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the latest conversation information.
            - user_id (int): The user ID.
            - conversation_number (int): The conversation number.
            - content (str): The conversation content.
            - timestamp (datetime): The timestamp of the conversation.

            or

            dict: A dictionary with a message indicating that no conversations were found for the user.

            or

            dict: A dictionary with an error message if an exception occurs.
    """
    try:
        user = db.query(User).filter(User.username == current_user.username).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        conversation = (
            db.query(Conversation)
            .filter(Conversation.user_id == user.id)
            .order_by(Conversation.timestamp.desc())
            .first()
        )

        if conversation:
            return {
                "user_id": conversation.user_id,
                "conversation_number": conversation.conversation_number,
                "content": conversation.content,
                "timestamp": conversation.timestamp,
            }
        else:
            return {"message": "No conversations found for the user"}
    except Exception as e:
        return {"error": str(e)}


@router.post("/update_conversation/")
async def update_conversation(
    input: Input,
    current_user: LoginUser = Security(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Update a conversation in the database.

    Args:
        input (Input): The input data for updating the conversation.
        current_user (LoginUser, optional): The currently logged-in user. Defaults to the current active user.
        db (Session, optional): The database session. Defaults to the dependency `get_db`.

    Returns:
        dict: A dictionary with the message "Conversation updated" if successful, or an error message if an exception occurs.
    """
    try:
        user = db.query(User).filter(User.username == current_user.username).first()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.conversation_number == input.conversation_number,
                Conversation.user_id == user.id,
            )
            .first()
        )

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Update conversation content
        conversation.content = input.content

        # Increment token numbers and check limits
        if input.prompt_token_number is not None:
            user.prompt_token_number += input.prompt_token_number

        if input.gen_token_number is not None:
            user.gen_token_number += input.gen_token_number

            # Disable user if gen_token_number exceeds token_limit
            if user.gen_token_number > user.token_limit:
                user.disabled = True
                user.token_limit = 0

        db.commit()

        return {"message": "Conversation updated"}
    except Exception as e:
        return {"error": str(e)}


@router.post("/update_conversation_name/")
async def update_conversation_name(
    input: Input,
    current_user: LoginUser = Security(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Updates the name of a conversation in the database.

    Parameters:
        - input (Input): The input object containing the conversation details.
        - current_user (LoginUser): The currently logged-in user.
        - db (Session): The database session.

    Returns:
        - dict: A dictionary containing the message indicating the success of the update, or an error message if an exception occurs.
    """
    try:
        # Check if the user exists by username
        user = db.query(User).filter(User.username == current_user.username).first()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Find the conversation for the user with the given conversation_number
        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.conversation_number == input.conversation_number,
                Conversation.user_id == user.id,
            )
            .first()
        )

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Update the name of the conversation
        conversation.conversation_name = input.conversation_name
        db.commit()

        return {"success": "Conversation name updated successfully"}

    except Exception as e:
        return {"error": str(e)}


@router.post("/retrieve_all_conversations/")
async def retrieve_all_conversations(
    current_user: LoginUser = Security(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Retrieve all conversations for a user.

    Parameters:
        - input: The input object containing the username.
        - current_user: The currently logged in user.
        - db: The database session.

    Returns:
        - A list of conversation numbers for the user.

    Raises:
        - Exception: If an error occurs during the retrieval process.
    """
    try:
        # Check if the user exists by username
        user = db.query(User).filter(User.username == current_user.username).first()

        if not user:
            return {"message": "User didn't exist"}  # User not found, return None

        # Retrieve all conversations for the user
        conversations = db.query(Conversation).filter(Conversation.user_id == user.id).all()

        # Extract conversation numbers
        conversation_numbers = [conv.conversation_number for conv in conversations]

        return {"conversation_numbers": conversation_numbers}

    except Exception as e:
        return {"error": str(e)}



@router.post("/get_user_conversations/")
async def get_user_conversations(
    input: Input,
    current_user: LoginUser = Security(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Retrieves the conversations for a given user.

    Parameters:
        - input: The input data containing the username.
        - current_user: The currently logged-in user.
        - db: The session to the database.

    Returns:
        A dictionary containing the conversations of the user. Each conversation is represented as a dictionary with the conversation number and name.
    """
    try:
        # Check if the user exists by username
        user = db.query(User).filter(User.username == current_user.username).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Retrieve conversation numbers and names for the user
        conversations = (
            db.query(Conversation.conversation_number, Conversation.conversation_name)
            .filter(Conversation.user_id == user.id)
            .all()
        )

        # Extract the conversation numbers and names from the result and store them in a list of dictionaries
        conversation_data = [{"number": cnv[0], "name": cnv[1]} for cnv in conversations]
        return {"conversations": conversation_data}

    except Exception as e:
        # Handle the exception here
        return {"error": str(e),"username":input.username}


