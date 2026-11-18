from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# 상품 카테고리별(식료품 vs 생필품) 기한 산정 및 사용자 소비 속도 개인화 엔진
class ConsumablePersonalizationEngine:
    def __init__(self, model_path: str = "consumable_predictor.pkl", alpha: float = 0.3):
        # 학습된 모델 아티팩트(.pkl) 로드
        artifact = joblib.load(model_path)
        self.model = artifact["model"]  # GradientBoostingRegressor 예측 모델
        self.model_columns = artifact["columns"]  # 모델 입력 피처 컬럼 순서
        self.base_specs = artifact["base_specs"]  # 카테고리별 기준 용량/단위 스펙
        # 등록되지 않은 세부 분류에 사용할 기본 폴백 스펙
        self.default_spec = artifact.get(
            "default_spec", {"base_amount": 1, "base_unit": "개", "base_days": 30}
        )
        self.alpha = alpha  # 누적 소비 이력 반영 시 k_factor 갱신 가중치

    @staticmethod
    def _normalize_amount(amount: float, unit: str, target_unit: str) -> float:
        """수량 단위를 기준 단위(ml, g 등)로 환산"""
        if unit == target_unit:
            return float(amount)
        if unit == "L" and target_unit == "ml":
            return float(amount) * 1000
        if unit == "ml" and target_unit == "L":
            return float(amount) / 1000
        if unit == "kg" and target_unit == "g":
            return float(amount) * 1000
        if unit == "g" and target_unit == "kg":
            return float(amount) / 1000
        return float(amount)

    def predict_consumable_days(
        self,
        subcategory: str,
        total_amount: float,
        unit: str,
        k_factor: float = 1.0,
        user_lifestyle_score: float = 1.0,
        base_date: datetime | None = None,
    ) -> dict:
        """생필품 1개 품목의 예상 소진 일수 및 예상 소진 날짜(product_date) 계산"""
        start_date = base_date or datetime.now()

        # 세부 분류명 매핑 (예: '물티슈' -> '일회용 위생용품', 매칭 없으면 그대로 사용)
        mapped_cat = "일회용 위생용품" if "물티슈" in subcategory else subcategory
        spec = self.base_specs.get(mapped_cat)

        # 기준 용량 대비 입력 수량 비율(amount_ratio) 계산
        if spec:
            normalized = self._normalize_amount(total_amount, unit, spec["base_unit"])
            amount_ratio = normalized / spec["base_amount"]
        else:
            normalized = self._normalize_amount(
                total_amount, unit, self.default_spec["base_unit"]
            )
            amount_ratio = normalized / self.default_spec["base_amount"]

        # 모델 입력용 피처 벡터 구성 (원-핫 카테고리 + 스케일링된 수량 + 라이프스타일 점수)
        row = {col: 0 for col in self.model_columns}
        cat_col = f"cat_{mapped_cat}"
        scaled_col = f"{mapped_cat}_scaled_amount"

        if cat_col in row:
            row[cat_col] = 1
        if scaled_col in row:
            row[scaled_col] = amount_ratio
        row["user_lifestyle_score"] = user_lifestyle_score

        # 학습 시점의 컬럼 순서에 맞춰 DataFrame 생성 후 기본 소진 일수 추론
        input_encoded = pd.DataFrame([row])[self.model_columns]
        raw_pred = float(self.model.predict(input_encoded)[0])
        base_days = max(1, round(raw_pred))  # 최소 1일 보장

        # 사용자 개인 소비 속도 계수(k_factor)를 반영한 최종 예상 소진 일수
        predicted_days = max(1, round(base_days * k_factor))

        return {
            "predicted_days": predicted_days,
            "product_date": (start_date + timedelta(days=predicted_days)).strftime(
                "%Y-%m-%d"
            ),
            "expiration_source": "ai_prediction",
            "allow_expiry_ocr": False,  # 생필품은 소비기한 OCR 불필요
        }

    def resolve_expiration(
        self,
        category: str,  # "생필품" | "식료품"
        subcategory: str,
        total_amount: float,
        unit: str,
        db_default_days: int = 14,  # 식료품용 DB 기본 임시기한(일)
        k_factor: float = 1.0,
        user_lifestyle_score: float = 1.0,
    ) -> dict:
        """대분류('생필품' / '식료품')에 따라 AI 예측 또는 DB 기본 기한을 반환"""
        if category == "생필품":
            return self.predict_consumable_days(
                subcategory=subcategory,
                total_amount=total_amount,
                unit=unit,
                k_factor=k_factor,
                user_lifestyle_score=user_lifestyle_score,
            )

        # 식료품: DB 기본 기한을 우선 산출하고, 패키지 소비기한 OCR 선택 허용
        safe_days = max(1, int(db_default_days))
        return {
            "predicted_days": None,
            "product_date": (datetime.now() + timedelta(days=safe_days)).strftime(
                "%Y-%m-%d"
            ),
            "expiration_source": "db_default",
            "allow_expiry_ocr": True,
        }

    def update_k_factor(
        self,
        current_k: float,
        base_days: int,
        actual_days: int,
        sample_count: int = 0,
    ) -> float:
        """실제 소진 일수(actual_days) 피드백을 바탕으로 사용자별 k_factor 갱신"""
        if base_days <= 0 or actual_days <= 0:
            return current_k

        # 극단적인 이상치 방지를 위해 관측 비율을 [0.4, 2.5] 범위로 클리핑
        observed_k = max(0.4, min(2.5, actual_days / base_days))

        # 첫 피드백(sample_count == 0)은 50:50 반영, 이후에는 지수이동평균(alpha) 적용
        if sample_count == 0:
            new_k = 0.5 * current_k + 0.5 * observed_k
        else:
            new_k = (1 - self.alpha) * current_k + self.alpha * observed_k
        return round(new_k, 3)


if __name__ == "__main__":
    # 엔진 인스턴스 초기화
    engine = ConsumablePersonalizationEngine("consumable_predictor.pkl")

    # 시연 영수증 예시 1: 물티슈(생필품 -> AI 소진일 예측)
    print("물티슈:", engine.resolve_expiration("생필품", "물티슈", 100, "매"))
    # 시연 영수증 예시 2: 과자(식료품 -> DB 임시기한 + OCR 허용)
    print("과자:", engine.resolve_expiration("식료품", "과자", 1, "봉"))
