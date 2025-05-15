from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.endpoints import video


app = FastAPI(
    title="YOLO Project API",
    description="YOLO Project Backend API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 특정 도메인만 허용하도록 수정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 라우터 등록
app.include_router(video.router, prefix="/api/video", tags=["video"])

@app.get("/")
async def root():
    return {"message": "Welcome to YOLO Project API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 