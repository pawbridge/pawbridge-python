import httpx
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image
import io

# 모델 초기화 (앱 시작 시 한 번만 로드)
_weights = MobileNet_V3_Large_Weights.DEFAULT
_model = mobilenet_v3_large(weights=_weights)
_model.classifier = torch.nn.Identity()  # 분류 레이어 제거 → 1280차원 벡터 출력
_model.eval()

# 이미지 전처리 파이프라인
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


async def extract_embedding_from_url(image_url: str) -> list[float] | None:
    """이미지 URL에서 이미지를 다운로드하고 임베딩 벡터 추출"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(image_url)
            response.raise_for_status()

        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        tensor = _transform(image).unsqueeze(0)  # (1, 3, 224, 224)

        with torch.no_grad():
            vector = _model(tensor).squeeze(0).tolist()  # (1280,) → list

        return vector

    except Exception:
        return None
