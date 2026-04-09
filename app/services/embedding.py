import httpx
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io

# DINOv2 ViT-S/14 모델 로드 (앱 시작 시 한 번만)
# - self-supervised 학습: 이미지 유사도 검색에 최적화 (분류 기반 MobileNetV3 대비 우수)
# - 출력: CLS 토큰 384-dim 벡터 (L2 정규화 후 코사인 유사도에 최적화)
# - CPU 전용 (torch +cpu 버전 설치 환경에서 자동 CPU 실행)
_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
_model.eval()

# 전처리 파이프라인 (DINOv2 권장: Resize→CenterCrop→정규화)
# - Resize(256) → CenterCrop(224): 피사체 중심 보존, 배경 노이즈 감소
# - 224 = 14 × 16 (ViT-S/14 패치 크기 14의 배수)
_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


async def extract_embedding_from_url(image_url: str) -> list[float] | None:
    """이미지 URL에서 DINOv2 임베딩 벡터 추출 (384-dim, L2 정규화)"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(image_url)
            response.raise_for_status()

        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        tensor = _transform(image).unsqueeze(0)  # (1, 3, 224, 224)

        with torch.no_grad():
            vector = _model(tensor)                   # (1, 384) CLS 토큰
            vector = F.normalize(vector, dim=-1)      # L2 정규화 → 코사인 유사도 일관성 보장
            return vector.squeeze(0).tolist()         # (384,) → list[float]

    except Exception:
        return None
