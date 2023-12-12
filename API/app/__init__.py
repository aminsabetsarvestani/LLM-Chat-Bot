from fastapi import FastAPI
from .routers import db_functions, login_router, inference, VDB_functions, arxiv_functions
from app.logging_config import setup_logger  # new import
logger = setup_logger() 
app = FastAPI()

app.include_router(login_router.router)
app.include_router(db_functions.router, prefix="/db_request", tags=["db_functions"])
app.include_router(inference.router, prefix="/llm_request", tags=["inference"])
app.include_router(VDB_functions.router, prefix="/vector_DB_request", tags=["VDB"])
app.include_router(arxiv_functions.router, prefix="/arxiv_search", tags=["arxiv"])