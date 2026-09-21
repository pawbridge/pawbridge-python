from fastapi import FastAPI
from app.routers import chatbot, lost_search
from app.services.lost_storage import backend

app = FastAPI(
    title="Pawbridge AI Service",
    description="유기동물 이미지 유사도 기반 추천 서비스",
    version="1.0.0"
)

# PostgreSQL cutover disables both legacy 384-dim inference and its batch endpoint.
# The authenticated recommendation route lives in app.lost_main, beside DINOv3.
if backend() == "elasticsearch":
    from app.routers import similarity
    app.include_router(similarity.router, prefix="/api/v1/animals", tags=["similarity"])
app.include_router(lost_search.router, prefix="/internal/animals", tags=["lost-search"])
app.include_router(chatbot.router, prefix="/internal/chatbot", tags=["chatbot"])


@app.get("/health")
def health_check():
    return {"status": "ok"}
