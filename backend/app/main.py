from fastapi import FastAPI
print("Starting FastAPI...")
from app.routes import router
print("Routes imported.")
app = FastAPI(
    title="Review Intelligence API",
    version="1.0"
)




app.include_router(router)