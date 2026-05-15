from fastapi import FastAPI
from app.routers import chatbot, similarity

app = FastAPI(
    title="Pawbridge AI Service",
    description="유기동물 이미지 유사도 기반 추천 서비스",
    version="1.0.0"
)

app.include_router(similarity.router, prefix="/api/v1/animals", tags=["similarity"])
app.include_router(chatbot.router, prefix="/internal/chatbot", tags=["chatbot"])


@app.get("/health")
def health_check():
    return {"status": "ok"}
