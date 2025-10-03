from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
        CORSMiddleware , 
        allow_origins=["https://localhost:5173"]
        allow_methods=["*"],
        allow_headers=["*"],
        )
 
