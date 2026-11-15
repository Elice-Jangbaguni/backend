
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

from fastapi import FastAPI, HTTPException, Form, File, UploadFile, Depends
from pydantic import BaseModel, Field, EmailStr
from pymysql.cursors import DictCursor

from database import get_connection
from pathlib import Path
from uuid import uuid4
from pwdlib import PasswordHash
from pymysql.err import IntegrityError

import shutil
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import os
import jwt
from fastapi.staticfiles import StaticFiles
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


app = FastAPI(title="장바구니 API")

from fastapi.middleware.cors import CORSMiddleware


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://jangbaguni-web.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path("uploads").mkdir(parents=True, exist_ok=True)

app.mount(
    "/uploads",
    StaticFiles(directory="uploads"),
    name="uploads"
)

password_hash = PasswordHash.recommended()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

if not JWT_SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY가 설정되지 않았습니다.")

JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60


# OCR 결과에서 등록할 상품 정보
class ProductItem(BaseModel):
    ocritem_id: int
    product_name: str = Field(min_length=1)
    category: str
    subcategory: Optional[str] = None
    product_date: Optional[date] = None


# 상품 등록 요청
class ProductRegisterRequest(BaseModel):
    receipt_id: int
    products: list[ProductItem] = Field(min_length=1)


def get_test_user_id(cursor):
    cursor.execute(
        "SELECT user_id FROM users WHERE username = %s",
        ("testuser",)
    )
    user = cursor.fetchone()

    if user is None:
        raise HTTPException(
            status_code=404,
            detail="테스트 사용자를 찾을 수 없습니다."
        )

    return user["user_id"]


security = HTTPBearer()


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM]
        )

        user_id = int(payload["sub"])

    except (jwt.InvalidTokenError, KeyError, ValueError, TypeError):
        raise HTTPException(
            status_code=401,
            detail="유효하지 않거나 만료된 토큰입니다.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    conn = None

    try:
        conn = get_connection()

        with conn.cursor(DictCursor) as cursor:
            cursor.execute("""
                SELECT user_id
                FROM users
                WHERE user_id = %s
            """, (user_id,))

            user = cursor.fetchone()

        if user is None:
            raise HTTPException(
                status_code=401,
                detail="존재하지 않는 사용자입니다."
            )

        return user_id

    finally:
        if conn is not None:
            conn.close()


@app.get("/")
def home():
    return {"message": "장바구니 백엔드 서버 실행 성공"}


@app.get("/products")
def get_products(
    user_id: int = Depends(get_current_user_id)
):
    conn = None

    try:
        conn = get_connection()

        with conn.cursor(DictCursor) as cursor:
            cursor.execute("""
                SELECT *
                FROM products
                WHERE user_id = %s
                ORDER BY product_id DESC
            """, (user_id,))

            products = cursor.fetchall()

        for product in products:
            if isinstance(product["total_amount"], Decimal):
                product["total_amount"] = float(
                    product["total_amount"]
                )

        return {
            "total": len(products),
            "page": 1,
            "limit": 20,
            "products": products
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="상품 목록 조회 중 오류가 발생했습니다."
        )

    finally:
        if conn is not None:
            conn.close()


@app.post("/products", status_code=201)
def register_products(request: ProductRegisterRequest, user_id: int = Depends(get_current_user_id)):
    conn = None

    try:
        conn = get_connection()

        with conn.cursor(DictCursor) as cursor:

            # 1. 영수증 소유자 확인
            cursor.execute("""
                SELECT receipt_id
                FROM receipts
                WHERE receipt_id = %s AND user_id = %s
            """, (request.receipt_id, user_id))

            if cursor.fetchone() is None:
                raise HTTPException(
                    status_code=404,
                    detail="해당 영수증을 찾을 수 없습니다."
                )

            # 2. 요청 내부의 중복 OCR 상품 확인
            item_ids = [
                item.ocritem_id for item in request.products
            ]

            if len(item_ids) != len(set(item_ids)):
                raise HTTPException(
                    status_code=400,
                    detail="중복된 OCR 상품이 포함되어 있습니다."
                )

            # 3. OCR 상품 확인 및 잠금
            placeholders = ",".join(["%s"] * len(item_ids))

            cursor.execute(f"""
                SELECT ocritem_id, product_id
                FROM receipt_items
                WHERE receipt_id = %s
                  AND ocritem_id IN ({placeholders})
                FOR UPDATE
            """, (request.receipt_id, *item_ids))

            ocr_items = {
                row["ocritem_id"]: row
                for row in cursor.fetchall()
            }

            if len(ocr_items) != len(item_ids):
                raise HTTPException(
                    status_code=404,
                    detail="OCR 상품 정보를 찾을 수 없습니다."
                )

            for item in request.products:
                if ocr_items[item.ocritem_id]["product_id"] is not None:
                    raise HTTPException(
                        status_code=409,
                        detail="이미 등록된 OCR 상품이 있습니다."
                    )

            registered_products = []

            # 4. 상품 등록
            for item in request.products:
                cursor.execute("""
                    INSERT INTO products (
                        user_id,
                        receipt_id,
                        product_name,
                        category,
                        subcategory,
                        product_date,
                        status
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_id,
                    request.receipt_id,
                    item.product_name,
                    item.category,
                    item.subcategory,
                    item.product_date,
                    "HEALTHY"
                ))

                product_id = cursor.lastrowid

                # 5. OCR 상품과 실제 상품 연결
                cursor.execute("""
                    UPDATE receipt_items
                    SET product_id = %s
                    WHERE ocritem_id = %s
                      AND receipt_id = %s
                """, (
                    product_id,
                    item.ocritem_id,
                    request.receipt_id
                ))

                registered_products.append({
                    "ocritem_id": item.ocritem_id,
                    "product_id": product_id,
                    "product_name": item.product_name
                })

        # 6. 모든 상품 등록이 성공하면 저장
        conn.commit()

        return {
            "receipt_id": request.receipt_id,
            "registered_count": len(registered_products),
            "products": registered_products
        }

    except HTTPException:
        if conn is not None:
            conn.rollback()
        raise

    except Exception:
        if conn is not None:
            conn.rollback()
        raise HTTPException(
            status_code=500,
            detail="상품 등록 중 오류가 발생했습니다."
        )

    finally:
        if conn is not None:
            conn.close()


@app.get("/api/receipts/{receipt_id}/items")
def get_receipt_items(receipt_id: int):
    conn = None

    try:
        conn = get_connection()

        with conn.cursor(DictCursor) as cursor:
            user_id = get_test_user_id(cursor)

            # 해당 사용자의 영수증인지 확인
            cursor.execute("""
                SELECT receipt_id
                FROM receipts
                WHERE receipt_id = %s
                  AND user_id = %s
            """, (receipt_id, user_id))

            if cursor.fetchone() is None:
                raise HTTPException(
                    status_code=404,
                    detail="해당 영수증을 찾을 수 없습니다."
                )

            # OCR 인식 상품 조회
            cursor.execute("""
                SELECT
                    ocritem_id,
                    product_id,
                    product_name,
                    category,
                    subcategory,
                    product_date
                FROM receipt_items
                WHERE receipt_id = %s
                ORDER BY ocritem_id
            """, (receipt_id,))

            items = cursor.fetchall()

        return {
            "receipt_id": receipt_id,
            "items": items
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="OCR 상품 조회 중 오류가 발생했습니다."
        )

    finally:
        if conn is not None:
            conn.close()


@app.delete("/api/products/{product_id}", status_code=204)
def delete_product(product_id: int,
    user_id: int = Depends(get_current_user_id)):
    conn = None

    try:
        conn = get_connection()

        with conn.cursor(DictCursor) as cursor:

            # 삭제할 상품이 현재 사용자의 것인지 확인
            cursor.execute("""
                SELECT product_id
                FROM products
                WHERE product_id = %s
                  AND user_id = %s
                FOR UPDATE
            """, (product_id, user_id))

            product = cursor.fetchone()

            if product is None:
                raise HTTPException(
                    status_code=404,
                    detail="상품을 찾을 수 없습니다."
                )

            # OCR 결과와 상품의 연결 해제
            cursor.execute("""
                UPDATE receipt_items
                SET product_id = NULL
                WHERE product_id = %s
            """, (product_id,))

            # 상품 삭제
            cursor.execute("""
                DELETE FROM products
                WHERE product_id = %s
                  AND user_id = %s
            """, (product_id, user_id))

        conn.commit()

        return None

    except HTTPException:
        if conn is not None:
            conn.rollback()
        raise

    except Exception:
        if conn is not None:
            conn.rollback()

        raise HTTPException(
            status_code=500,
            detail="상품 삭제 중 오류가 발생했습니다."
        )

    finally:
        if conn is not None:
            conn.close()



@app.patch("/api/products/{product_id}/deplete")
def deplete_product(product_id: int, user_id: int = Depends(get_current_user_id)):
    conn = None

    try:
        conn = get_connection()

        with conn.cursor(DictCursor) as cursor:

            # 1. 상품 확인
            cursor.execute("""
                SELECT
                    product_id,
                    product_name,
                    category,
                    status,
                    depleted_date
                FROM products
                WHERE product_id = %s
                  AND user_id = %s
                FOR UPDATE
            """, (product_id, user_id))

            product = cursor.fetchone()

            if product is None:
                raise HTTPException(
                    status_code=404,
                    detail="상품을 찾을 수 없습니다."
                )

            # 2. 생필품인지 확인
            if product["category"] != "daily":
                raise HTTPException(
                    status_code=400,
                    detail="생필품만 소진 처리할 수 있습니다."
                )

            # 3. 이미 소진된 상품인지 확인
            if product["status"] == "DEPLETED":
                raise HTTPException(
                    status_code=409,
                    detail="이미 소진 처리된 상품입니다."
                )

            # 4. 소진 상태 및 날짜 변경
            today = date.today()

            cursor.execute("""
                UPDATE products
                SET status = %s,
                    depleted_date = %s
                WHERE product_id = %s
                  AND user_id = %s
            """, (
                "DEPLETED",
                today,
                product_id,
                user_id
            ))

        conn.commit()

        return {
            "product_id": product_id,
            "product_name": product["product_name"],
            "status": "DEPLETED",
            "depleted_date": today.isoformat()
        }

    except HTTPException:
        if conn is not None:
            conn.rollback()
        raise

    except Exception:
        if conn is not None:
            conn.rollback()

        raise HTTPException(
            status_code=500,
            detail="상품 소진 처리 중 오류가 발생했습니다."
        )

    finally:
        if conn is not None:
            conn.close()


@app.get("/api/group_buy")
def get_group_buy_list(region: str):
    conn = None

    try:
        conn = get_connection()

        with conn.cursor(DictCursor) as cursor:
            cursor.execute("""
                SELECT
                    f.buy_id,
                    f.user_id,
                    f.region,
                    f.product_name,
                    f.product_image_path,
                    f.price,
                    f.place,
                    f.number_of_people,
                    f.description,
                    f.uploaded_date,
                    f.status,
                    (
                        SELECT COUNT(*)
                        FROM group_buy AS p
                        WHERE p.buy_id = f.buy_id
                    ) AS current_people
                FROM group_buy_feed AS f
                WHERE f.region = %s
                ORDER BY f.uploaded_date DESC, f.buy_id DESC
            """, (region,))

            posts = cursor.fetchall()

        return {
            "region": region,
            "total": len(posts),
            "posts": posts
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="공동구매 목록 조회 중 오류가 발생했습니다."
        )

    finally:
        if conn is not None:
            conn.close()

UPLOAD_DIR = Path("uploads/group-buy")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/api/group_buy", status_code=201)
def create_group_buy(
    region: str = Form(...),
    product_name: str = Form(...),
    price: int = Form(..., gt=0),
    place: str = Form(...),
    number_of_people: int = Form(..., ge=2),
    description: str = Form(""),
    product_image: UploadFile | None = File(None),
    user_id: int = Depends(get_current_user_id)
):
    conn = None
    image_path = None
    saved_file = None

    try:
        # 1. 이미지 저장
        if product_image is not None:
            allowed_types = {
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/webp": ".webp"
            }

            extension = allowed_types.get(
                product_image.content_type
            )

            if extension is None:
                raise HTTPException(
                    status_code=400,
                    detail="JPG, PNG, WEBP 이미지만 업로드할 수 있습니다."
                )

            filename = f"{uuid4().hex}{extension}"
            saved_file = UPLOAD_DIR / filename

            # 최대 5MB 제한
            image_data = product_image.file.read(5 * 1024 * 1024 + 1)

            if len(image_data) > 5 * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail="이미지는 최대 5MB까지 업로드할 수 있습니다."
                )

            with open(saved_file, "wb") as f:
                f.write(image_data)

            image_path = f"/uploads/group-buy/{filename}"

        # 2. DB 연결
        conn = get_connection()

        with conn.cursor(DictCursor) as cursor:

            # 3. 공동구매 게시글 저장
            cursor.execute("""
                INSERT INTO group_buy_feed (
                    user_id,
                    region,
                    product_name,
                    product_image_path,
                    price,
                    place,
                    number_of_people,
                    description,
                    status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                user_id,
                region,
                product_name,
                image_path,
                price,
                place,
                number_of_people,
                description,
                "recruiting"
            ))

            buy_id = cursor.lastrowid

            # 4. 작성자도 참여자로 등록
            cursor.execute("""
                INSERT INTO group_buy (
                    buy_id,
                    user_id
                )
                VALUES (%s, %s)
            """, (buy_id, user_id))

        conn.commit()

        return {
            "buy_id": buy_id,
            "user_id": user_id,
            "region": region,
            "product_name": product_name,
            "product_image_path": image_path,
            "price": price,
            "place": place,
            "number_of_people": number_of_people,
            "current_people": 1,
            "description": description,
            "status": "recruiting"
        }

    except HTTPException:
        if conn is not None:
            conn.rollback()
        if saved_file is not None:
            saved_file.unlink(missing_ok=True)
        raise

    except Exception:
        if conn is not None:
            conn.rollback()
        if saved_file is not None:
            saved_file.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="공동구매 글 작성 중 오류가 발생했습니다."
        )

    finally:
        if conn is not None:
            conn.close()
        if product_image is not None:
            product_image.file.close()


@app.get("/api/group_buy/{buy_id}")
def get_group_buy_detail(buy_id: int,user_id: int = Depends(get_current_user_id)
):
    conn = None

    try:
        conn = get_connection()

        with conn.cursor(DictCursor) as cursor:

            # 공동구매 글 상세 정보 조회
            cursor.execute("""
                SELECT
                    f.buy_id,
                    f.user_id,
                    f.region,
                    f.product_name,
                    f.product_image_path,
                    f.price,
                    f.place,
                    f.number_of_people,
                    f.description,
                    f.uploaded_date,
                    f.status,

                    (
                        SELECT COUNT(*)
                        FROM group_buy AS p
                        WHERE p.buy_id = f.buy_id
                    ) AS current_people,

                    EXISTS (
                        SELECT 1
                        FROM group_buy AS p
                        WHERE p.buy_id = f.buy_id
                          AND p.user_id = %s
                    ) AS is_participating

                FROM group_buy_feed AS f
                WHERE f.buy_id = %s
            """, (user_id, buy_id))

            post = cursor.fetchone()

            if post is None:
                raise HTTPException(
                    status_code=404,
                    detail="공동구매 글을 찾을 수 없습니다."
                )

            # 작성자 여부
            post["is_owner"] = (
                post["user_id"] == user_id
            )

            # 참여 여부를 boolean으로 변환
            post["is_participating"] = bool(
                post["is_participating"]
            )

        return post

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="공동구매 상세 조회 중 오류가 발생했습니다."
        )

    finally:
        if conn is not None:
            conn.close()



@app.put("/api/group_buy/{buy_id}")
def update_group_buy(
    buy_id: int,
    region: str = Form(...),
    product_name: str = Form(...),
    price: int = Form(..., gt=0),
    place: str = Form(...),
    number_of_people: int = Form(..., ge=2),
    description: str = Form(""),
    product_image: UploadFile | None = File(None),
    user_id: int = Depends(get_current_user_id)

):
    conn = None
    saved_file = None

    try:
        conn = get_connection()

        with conn.cursor(DictCursor) as cursor:

            # 1. 게시글 확인
            cursor.execute("""
                SELECT *
                FROM group_buy_feed
                WHERE buy_id = %s
                FOR UPDATE
            """, (buy_id,))

            post = cursor.fetchone()

            if post is None:
                raise HTTPException(
                    status_code=404,
                    detail="공동구매 글을 찾을 수 없습니다."
                )

            # 2. 작성자 확인
            if post["user_id"] != user_id:
                raise HTTPException(
                    status_code=403,
                    detail="작성자만 수정할 수 있습니다."
                )

            # 3. 현재 참여 인원 확인
            cursor.execute("""
                SELECT COUNT(*) AS current_people
                FROM group_buy
                WHERE buy_id = %s
            """, (buy_id,))

            current_people = cursor.fetchone()["current_people"]

            if number_of_people < current_people:
                raise HTTPException(
                    status_code=400,
                    detail="모집 인원을 현재 참여 인원보다 적게 설정할 수 없습니다."
                )

            # 4. 이미지 처리
            image_path = post["product_image_path"]

            if product_image is not None:
                allowed_types = {
                    "image/jpeg": ".jpg",
                    "image/png": ".png",
                    "image/webp": ".webp"
                }

                extension = allowed_types.get(
                    product_image.content_type
                )

                if extension is None:
                    raise HTTPException(
                        status_code=400,
                        detail="JPG, PNG, WEBP 이미지만 업로드할 수 있습니다."
                    )

                image_data = product_image.file.read(
                    5 * 1024 * 1024 + 1
                )

                if len(image_data) > 5 * 1024 * 1024:
                    raise HTTPException(
                        status_code=413,
                        detail="이미지는 최대 5MB까지 업로드할 수 있습니다."
                    )

                filename = f"{uuid4().hex}{extension}"
                saved_file = UPLOAD_DIR / filename

                with open(saved_file, "wb") as f:
                    f.write(image_data)

                image_path = f"/uploads/group-buy/{filename}"

            # 5. 게시글 수정
            cursor.execute("""
                UPDATE group_buy_feed
                SET
                    region = %s,
                    product_name = %s,
                    product_image_path = %s,
                    price = %s,
                    place = %s,
                    number_of_people = %s,
                    description = %s
                WHERE buy_id = %s
                  AND user_id = %s
            """, (
                region,
                product_name,
                image_path,
                price,
                place,
                number_of_people,
                description,
                buy_id,
                user_id
            ))

        conn.commit()

        return {
            "buy_id": buy_id,
            "user_id": user_id,
            "region": region,
            "product_name": product_name,
            "product_image_path": image_path,
            "price": price,
            "place": place,
            "number_of_people": number_of_people,
            "current_people": current_people,
            "description": description,
            "uploaded_date": post["uploaded_date"],
            "status": post["status"]
        }

    except HTTPException:
        if conn is not None:
            conn.rollback()

        if saved_file is not None:
            saved_file.unlink(missing_ok=True)

        raise

    except Exception:
        if conn is not None:
            conn.rollback()

        if saved_file is not None:
            saved_file.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="공동구매 글 수정 중 오류가 발생했습니다."
        )

    finally:
        if conn is not None:
            conn.close()

        if product_image is not None:
            product_image.file.close()

@app.get("/api/group_buy/participants")
def get_my_group_buys(    user_id: int = Depends(get_current_user_id)):
    conn = None

    try:
        conn = get_connection()

        with conn.cursor(DictCursor) as cursor:

            # 내가 참여 중인 공동구매 조회
            cursor.execute("""
                SELECT
                    f.buy_id,
                    f.user_id,
                    f.region,
                    f.product_name,
                    f.product_image_path,
                    f.price,
                    f.place,
                    f.number_of_people,
                    f.description,
                    f.uploaded_date,
                    f.status,

                    (
                        SELECT COUNT(*)
                        FROM group_buy AS p2
                        WHERE p2.buy_id = f.buy_id
                    ) AS current_people,

                    CASE
                        WHEN f.user_id = %s THEN TRUE
                        ELSE FALSE
                    END AS is_owner

                FROM group_buy AS p
                JOIN group_buy_feed AS f
                    ON p.buy_id = f.buy_id

                WHERE p.user_id = %s

                ORDER BY f.uploaded_date DESC, f.buy_id DESC
            """, (user_id, user_id))

            posts = cursor.fetchall()

        # MySQL의 TRUE/FALSE 결과를 Python bool로 변환
        for post in posts:
            post["is_owner"] = bool(post["is_owner"])

        return {
            "total": len(posts),
            "posts": posts
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="참여 중인 공동구매 조회 중 오류가 발생했습니다."
        )

    finally:
        if conn is not None:
            conn.close()

@app.post("/api/group_buy/{buy_id}/participants", status_code=201)
def join_group_buy(buy_id: int,    user_id: int = Depends(get_current_user_id)
):
    conn = None

    try:
        conn = get_connection()

        with conn.cursor(DictCursor) as cursor:

            # 1. 공동구매 게시글 확인 및 잠금
            cursor.execute("""
                SELECT
                    buy_id,
                    user_id,
                    number_of_people,
                    status
                FROM group_buy_feed
                WHERE buy_id = %s
                FOR UPDATE
            """, (buy_id,))

            post = cursor.fetchone()

            if post is None:
                raise HTTPException(
                    status_code=404,
                    detail="공동구매 글을 찾을 수 없습니다."
                )

            # 2. 중복 참여 확인
            cursor.execute("""
                SELECT participant_id
                FROM group_buy
                WHERE buy_id = %s
                  AND user_id = %s
            """, (buy_id, user_id))

            if cursor.fetchone() is not None:
                raise HTTPException(
                    status_code=409,
                    detail="이미 참여 중인 공동구매입니다."
                )

            # 3. 모집 상태 확인
            if post["status"] != "recruiting":
                raise HTTPException(
                    status_code=400,
                    detail="모집이 종료된 공동구매입니다."
                )

            # 4. 현재 참여 인원 계산
            cursor.execute("""
                SELECT COUNT(*) AS current_people
                FROM group_buy
                WHERE buy_id = %s
            """, (buy_id,))

            current_people = cursor.fetchone()["current_people"]

            if current_people >= post["number_of_people"]:
                raise HTTPException(
                    status_code=400,
                    detail="모집 인원이 가득 찼습니다."
                )

            # 5. 공동구매 참여자 등록
            cursor.execute("""
                INSERT INTO group_buy (
                    buy_id,
                    user_id
                )
                VALUES (%s, %s)
            """, (buy_id, user_id))

            participant_id = cursor.lastrowid
            current_people += 1

            # 6. 모집 인원에 도달하면 모집 종료
            status = post["status"]

            if current_people >= post["number_of_people"]:
                cursor.execute("""
                    UPDATE group_buy_feed
                    SET status = 'closed'
                    WHERE buy_id = %s
                """, (buy_id,))

                status = "closed"

        conn.commit()

        return {
            "participant_id": participant_id,
            "buy_id": buy_id,
            "user_id": user_id,
            "current_people": current_people,
            "number_of_people": post["number_of_people"],
            "status": status
        }

    except HTTPException:
        if conn is not None:
            conn.rollback()
        raise

    except Exception:
        if conn is not None:
            conn.rollback()

        raise HTTPException(
            status_code=500,
            detail="공동구매 참여 중 오류가 발생했습니다."
        )

    finally:
        if conn is not None:
            conn.close()


@app.get("/api/group_buy/{buy_id}/messages")
def get_chat_messages(buy_id: int,user_id: int = Depends(get_current_user_id)):
    conn = None

    try:
        conn = get_connection()

        with conn.cursor(DictCursor) as cursor:

            # 1. 공동구매 글 존재 여부 확인
            cursor.execute("""
                SELECT buy_id
                FROM group_buy_feed
                WHERE buy_id = %s
            """, (buy_id,))

            if cursor.fetchone() is None:
                raise HTTPException(
                    status_code=404,
                    detail="공동구매 글을 찾을 수 없습니다."
                )

            # 2. 채팅방 참여 여부 확인
            cursor.execute("""
                SELECT participant_id
                FROM group_buy
                WHERE buy_id = %s
                  AND user_id = %s
            """, (buy_id, user_id))

            if cursor.fetchone() is None:
                raise HTTPException(
                    status_code=403,
                    detail="공동구매 참여자만 채팅을 조회할 수 있습니다."
                )

            # 3. 채팅 메시지 조회
            cursor.execute("""
                SELECT
                    message_id,
                    buy_id,
                    user_id,
                    message,
                    image_path,
                    sent_time
                FROM group_buy_chat
                WHERE buy_id = %s
                ORDER BY sent_time ASC, message_id ASC
            """, (buy_id,))

            messages = cursor.fetchall()

        return {
            "buy_id": buy_id,
            "total": len(messages),
            "messages": messages
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="채팅 내역 조회 중 오류가 발생했습니다."
        )

    finally:
        if conn is not None:
            conn.close()


CHAT_UPLOAD_DIR = Path("uploads/chat")
CHAT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/api/group_buy/{buy_id}/messages", status_code=201)
def send_chat_message(
    buy_id: int,
    message: str | None = Form(None),
    image: UploadFile | None = File(None),
    user_id: int = Depends(get_current_user_id)
):
    conn = None
    saved_file = None

    try:
        # 1. 메시지 내용 확인
        message_text = message.strip() if message else None

        if not message_text and image is None:
            raise HTTPException(
                status_code=400,
                detail="메시지 또는 이미지를 입력해야 합니다."
            )

        conn = get_connection()

        with conn.cursor(DictCursor) as cursor:

            # 2. 공동구매 글 확인
            cursor.execute("""
                SELECT buy_id
                FROM group_buy_feed
                WHERE buy_id = %s
            """, (buy_id,))

            if cursor.fetchone() is None:
                raise HTTPException(
                    status_code=404,
                    detail="공동구매 글을 찾을 수 없습니다."
                )

            # 3. 참여자 확인
            cursor.execute("""
                SELECT participant_id
                FROM group_buy
                WHERE buy_id = %s
                  AND user_id = %s
            """, (buy_id, user_id))

            if cursor.fetchone() is None:
                raise HTTPException(
                    status_code=403,
                    detail="공동구매 참여자만 메시지를 보낼 수 있습니다."
                )

            # 4. 이미지 저장
            image_path = None

            if image is not None:
                allowed_types = {
                    "image/jpeg": ".jpg",
                    "image/png": ".png",
                    "image/webp": ".webp"
                }

                extension = allowed_types.get(image.content_type)

                if extension is None:
                    raise HTTPException(
                        status_code=400,
                        detail="JPG, PNG, WEBP 이미지만 전송할 수 있습니다."
                    )

                image_data = image.file.read(5 * 1024 * 1024 + 1)

                if len(image_data) > 5 * 1024 * 1024:
                    raise HTTPException(
                        status_code=413,
                        detail="이미지는 최대 5MB까지 전송할 수 있습니다."
                    )

                filename = f"{uuid4().hex}{extension}"
                saved_file = CHAT_UPLOAD_DIR / filename

                with open(saved_file, "wb") as f:
                    f.write(image_data)

                image_path = f"/uploads/chat/{filename}"

            # 5. 채팅 메시지 저장
            cursor.execute("""
                INSERT INTO group_buy_chat (
                    buy_id,
                    user_id,
                    message,
                    image_path
                )
                VALUES (%s, %s, %s, %s)
            """, (
                buy_id,
                user_id,
                message_text,
                image_path
            ))

            message_id = cursor.lastrowid

            # 6. 저장된 메시지 조회
            cursor.execute("""
                SELECT
                    message_id,
                    buy_id,
                    user_id,
                    message,
                    image_path,
                    sent_time
                FROM group_buy_chat
                WHERE message_id = %s
            """, (message_id,))

            saved_message = cursor.fetchone()

        conn.commit()

        return saved_message

    except HTTPException:
        if conn is not None:
            conn.rollback()

        if saved_file is not None:
            saved_file.unlink(missing_ok=True)

        raise

    except Exception:
        if conn is not None:
            conn.rollback()

        if saved_file is not None:
            saved_file.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="채팅 메시지 전송 중 오류가 발생했습니다."
        )

    finally:
        if conn is not None:
            conn.close()

        if image is not None:
            image.file.close()


from collections import defaultdict

# 채팅방별 WebSocket 연결 목록
chat_rooms = defaultdict(list)

async def broadcast_message(buy_id: int, data: dict):
    disconnected = []

    for websocket in list(chat_rooms[buy_id]):
        try:
            await websocket.send_json(data)
        except Exception:
            disconnected.append(websocket)

    for websocket in disconnected:
        if websocket in chat_rooms[buy_id]:
            chat_rooms[buy_id].remove(websocket)



@app.websocket("/ws/group_buy/{buy_id}")
async def group_buy_websocket(
    websocket: WebSocket,
    buy_id: int
):
    # 1. WebSocket 연결 수락
    await websocket.accept()

    user_id = None

    try:
        # 2. 첫 메시지에서 JWT 수신
        auth_data = await asyncio.wait_for(
            websocket.receive_json(),
            timeout=10
        )

        if (
            not isinstance(auth_data, dict)
            or auth_data.get("type") != "auth"
            or not isinstance(auth_data.get("token"), str)
        ):
            await websocket.close(code=1008)
            return

        user_id = verify_websocket_token(
            auth_data["token"]
        )

        if user_id is None:
            await websocket.close(code=1008)
            return

        # 3. DB에서 사용자 및 참여 여부 확인
        conn = None

        try:
            conn = get_connection()

            with conn.cursor(DictCursor) as cursor:
                cursor.execute("""
                    SELECT u.user_id
                    FROM users AS u
                    JOIN group_buy AS p
                        ON u.user_id = p.user_id
                    WHERE u.user_id = %s
                      AND p.buy_id = %s
                """, (user_id, buy_id))

                participant = cursor.fetchone()

            if participant is None:
                await websocket.close(code=1008)
                return

        finally:
            if conn is not None:
                conn.close()

        # 4. 인증 성공
        chat_rooms[buy_id].append(websocket)

        await websocket.send_json({
            "type": "auth_success",
            "buy_id": buy_id,
            "user_id": user_id
        })

        # 5. 메시지 수신
        while True:
            data = await websocket.receive_json()

            if not isinstance(data, dict):
                await websocket.send_json({
                    "type": "error",
                    "detail": "올바른 메시지 형식이 아닙니다."
                })
                continue

            if data.get("type") != "message":
                await websocket.send_json({
                    "type": "error",
                    "detail": "지원하지 않는 메시지 유형입니다."
                })
                continue

            message = data.get("message")

            if not isinstance(message, str) or not message.strip():
                await websocket.send_json({
                    "type": "error",
                    "detail": "메시지를 입력해야 합니다."
                })
                continue

            message = message.strip()

            if len(message) > 5000:
                await websocket.send_json({
                    "type": "error",
                    "detail": "메시지는 5000자 이하여야 합니다."
                })
                continue

            # 6. DB에 메시지 저장
            db = None
            participant_removed = False

            try:
                db = get_connection()

                with db.cursor(DictCursor) as cursor:
                    cursor.execute("""
                        SELECT participant_id
                        FROM group_buy
                        WHERE buy_id = %s
                          AND user_id = %s
                    """, (buy_id, user_id))

                    if cursor.fetchone() is None:
                        participant_removed = True
                    else:
                        cursor.execute("""
                            INSERT INTO group_buy_chat (
                                buy_id,
                                user_id,
                                message,
                                image_path
                            )
                            VALUES (%s, %s, %s, NULL)
                        """, (buy_id, user_id, message))

                        message_id = cursor.lastrowid

                        cursor.execute("""
                            SELECT
                                message_id,
                                buy_id,
                                user_id,
                                message,
                                image_path,
                                sent_time
                            FROM group_buy_chat
                            WHERE message_id = %s
                        """, (message_id,))

                        saved_message = cursor.fetchone()

                if not participant_removed:
                    db.commit()

            except Exception:
                if db is not None:
                    db.rollback()

                await websocket.send_json({
                    "type": "error",
                    "detail": "메시지 저장에 실패했습니다."
                })
                continue

            finally:
                if db is not None:
                    db.close()

            if participant_removed:
                await websocket.close(code=1008)
                break

            # 7. 같은 채팅방에 실시간 전송
            await broadcast_message(buy_id, {
                "type": "new_message",
                "data": {
                    **saved_message,
                    "sent_time": saved_message[
                        "sent_time"
                    ].isoformat()
                }
            })

    except WebSocketDisconnect:
        pass

    except asyncio.TimeoutError:
        await websocket.close(code=1008)

    except Exception:
        try:
            await websocket.close(code=1011)
        except Exception:
            pass

    finally:
        if buy_id in chat_rooms:
            if websocket in chat_rooms[buy_id]:
                chat_rooms[buy_id].remove(websocket)

            if not chat_rooms[buy_id]:
                del chat_rooms[buy_id]



@app.delete("/api/group_buy/{buy_id}/participants/me")
def leave_group_buy(buy_id: int, user_id: int = Depends(get_current_user_id)):
    conn = None

    try:
        conn = get_connection()

        with conn.cursor(DictCursor) as cursor:

            # 1. 공동구매 글 확인 및 잠금
            cursor.execute("""
                SELECT
                    buy_id,
                    user_id,
                    number_of_people,
                    status
                FROM group_buy_feed
                WHERE buy_id = %s
                FOR UPDATE
            """, (buy_id,))

            post = cursor.fetchone()

            if post is None:
                raise HTTPException(
                    status_code=404,
                    detail="공동구매 글을 찾을 수 없습니다."
                )

            # 2. 작성자는 나가기 불가
            if post["user_id"] == user_id:
                raise HTTPException(
                    status_code=403,
                    detail="공동구매 작성자는 나갈 수 없습니다."
                )

            # 3. 참여 여부 확인
            cursor.execute("""
                SELECT participant_id
                FROM group_buy
                WHERE buy_id = %s
                  AND user_id = %s
            """, (buy_id, user_id))

            participant = cursor.fetchone()

            if participant is None:
                raise HTTPException(
                    status_code=404,
                    detail="참여 중인 공동구매가 아닙니다."
                )

            # 4. 참여 기록 삭제
            cursor.execute("""
                DELETE FROM group_buy
                WHERE buy_id = %s
                  AND user_id = %s
            """, (buy_id, user_id))

            # 5. 남은 참여 인원 계산
            cursor.execute("""
                SELECT COUNT(*) AS current_people
                FROM group_buy
                WHERE buy_id = %s
            """, (buy_id,))

            current_people = cursor.fetchone()["current_people"]

            # 6. 인원에 여유가 생기면 모집 재개
            status = post["status"]

            if (
                status == "closed"
                and current_people < post["number_of_people"]
            ):
                cursor.execute("""
                    UPDATE group_buy_feed
                    SET status = 'recruiting'
                    WHERE buy_id = %s
                """, (buy_id,))

                status = "recruiting"

        conn.commit()

        return {
            "buy_id": buy_id,
            "user_id": user_id,
            "message": "공동구매에서 나갔습니다.",
            "current_people": current_people,
            "number_of_people": post["number_of_people"],
            "status": status
        }

    except HTTPException:
        if conn is not None:
            conn.rollback()
        raise

    except Exception:
        if conn is not None:
            conn.rollback()

        raise HTTPException(
            status_code=500,
            detail="공동구매 나가기 중 오류가 발생했습니다."
        )

    finally:
        if conn is not None:
            conn.close()


class DepletionPredictionRequest(BaseModel):
    product_id: int


# 임시 기준 사용 기간 (일)
DEFAULT_USAGE_DAYS = {
    "세제": 30,
    "샴푸": 45,
    "휴지": 20,
    "치약": 40
}


@app.post("/api/products/predict-depletion")
def predict_depletion(request: DepletionPredictionRequest,
    user_id: int = Depends(get_current_user_id)):
    conn = None

    try:
        conn = get_connection()

        with conn.cursor(DictCursor) as cursor:

            # 1. 상품 조회
            cursor.execute("""
                SELECT
                    product_id,
                    user_id,
                    product_name,
                    category,
                    subcategory,
                    purchased_date,
                    total_amount,
                    unit,
                    product_date,
                    k_factor,
                    status
                FROM products
                WHERE product_id = %s
                  AND user_id = %s
            """, (request.product_id, user_id))

            product = cursor.fetchone()

            if product is None:
                raise HTTPException(
                    status_code=404,
                    detail="상품을 찾을 수 없습니다."
                )

            # 2. 생필품 여부 확인
            if product["category"] != "daily":
                raise HTTPException(
                    status_code=400,
                    detail="생필품만 소진 시점을 예측할 수 있습니다."
                )

            if product["status"] == "DEPLETED":
                raise HTTPException(
                    status_code=400,
                    detail="이미 소진 처리된 상품입니다."
                )

            # 3. 구매일 확인
            purchased_date = product["purchased_date"]

            if purchased_date is None:
                raise HTTPException(
                    status_code=400,
                    detail="구매일 정보가 없습니다."
                )

            if isinstance(purchased_date, datetime):
                purchased_date = purchased_date.date()

            # 4. 상품별 기준 사용 기간
            subcategory = product["subcategory"]

            base_days = DEFAULT_USAGE_DAYS.get(
                subcategory,
                30
            )

            # 5. 사용자별 보정 계수 조회
            cursor.execute("""
                SELECT
                    k_factor,
                    sample_count
                FROM user_k_factors
                WHERE user_id = %s
                  AND category = %s
            """, (user_id, subcategory))

            user_factor = cursor.fetchone()

            if user_factor is not None:
                k_factor = float(user_factor["k_factor"])
            elif product["k_factor"] is not None:
                k_factor = float(product["k_factor"])
            else:
                k_factor = 1.0

            if k_factor <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="유효하지 않은 보정 계수입니다."
                )

            # 6. 예상 사용 기간 계산
            predicted_days = max(
                1,
                round(base_days * k_factor)
            )

            # 7. 예상 소진일 계산
            predicted_date = (
                purchased_date
                + timedelta(days=predicted_days)
            )

            remaining_days = (
                predicted_date - date.today()
            ).days

            # 8. 상품 상태 계산
            if remaining_days < 0:
                status = "EXPIRED"
            elif remaining_days <= 7:
                status = "CRITICAL"
            else:
                status = "HEALTHY"

            # 9. 예측 결과 DB 저장
            cursor.execute("""
                UPDATE products
                SET
                    product_date = %s,
                    k_factor = %s,
                    status = %s
                WHERE product_id = %s
                  AND user_id = %s
            """, (
                predicted_date,
                k_factor,
                status,
                request.product_id,
                user_id
            ))

        conn.commit()

        return {
            "product_id": request.product_id,
            "product_name": product["product_name"],
            "base_days": base_days,
            "k_factor": k_factor,
            "predicted_days": predicted_days,
            "predicted_depletion_date": predicted_date.isoformat(),
            "remaining_days": remaining_days,
            "status": status
        }

    except HTTPException:
        if conn is not None:
            conn.rollback()
        raise

    except Exception:
        if conn is not None:
            conn.rollback()

        raise HTTPException(
            status_code=500,
            detail="소진 시점 예측 중 오류가 발생했습니다."
        )

    finally:
        if conn is not None:
            conn.close()


class SignupRequest(BaseModel):
    username: str = Field(min_length=4, max_length=50)
    password: str = Field(min_length=8, max_length=128)
    birth: date
    gender: str
    email: Optional[EmailStr] = None


@app.post("/api/auth/signup", status_code=201)
def signup(request: SignupRequest):
    conn = None

    try:
        conn = get_connection()

        with conn.cursor(DictCursor) as cursor:

            # 1. 아이디 중복 확인
            cursor.execute("""
                SELECT user_id
                FROM users
                WHERE username = %s
            """, (request.username,))

            if cursor.fetchone() is not None:
                raise HTTPException(
                    status_code=409,
                    detail="이미 사용 중인 아이디입니다."
                )

            # 2. 이메일 중복 확인
            if request.email is not None:
                cursor.execute("""
                    SELECT user_id
                    FROM users
                    WHERE email = %s
                """, (request.email,))

                if cursor.fetchone() is not None:
                    raise HTTPException(
                        status_code=409,
                        detail="이미 사용 중인 이메일입니다."
                    )

            # 3. 비밀번호 해시 생성
            hashed_password = password_hash.hash(
                request.password
            )

            # 4. 사용자 정보 저장
            cursor.execute("""
                INSERT INTO users (
                    username,
                    password_hash,
                    birth,
                    gender,
                    email
                )
                VALUES (%s, %s, %s, %s, %s)
            """, (
                request.username,
                hashed_password,
                request.birth,
                request.gender,
                request.email
            ))

            user_id = cursor.lastrowid

        conn.commit()

        return {
            "user_id": user_id,
            "username": request.username,
            "email": request.email,
            "message": "회원가입이 완료되었습니다."
        }

    except HTTPException:
        if conn is not None:
            conn.rollback()
        raise

    except IntegrityError:
        if conn is not None:
            conn.rollback()

        raise HTTPException(
            status_code=409,
            detail="이미 등록된 아이디 또는 이메일입니다."
        )

    except Exception as e:
        print("회원가입 오류:", repr(e))
        raise HTTPException(
            status_code=500,
            detail="회원가입 중 오류가 발생했습니다."
        )

    finally:
        if conn is not None:
            conn.close()


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
def login(request: LoginRequest):
    conn = None

    try:
        conn = get_connection()

        with conn.cursor(DictCursor) as cursor:

            # 1. 아이디로 사용자 조회
            cursor.execute("""
                SELECT
                    user_id,
                    username,
                    password_hash
                FROM users
                WHERE username = %s
            """, (request.username,))

            user = cursor.fetchone()

        # 2. 아이디 및 비밀번호 검증
        if user is None:
            raise HTTPException(
                status_code=401,
                detail="아이디 또는 비밀번호가 올바르지 않습니다."
            )

        try:
            password_valid = password_hash.verify(
                request.password,
                user["password_hash"]
            )
        except Exception:
            password_valid = False

        if not password_valid:
            raise HTTPException(
                status_code=401,
                detail="아이디 또는 비밀번호가 올바르지 않습니다."
            )

        # 3. JWT 만료 시간 설정
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        )

        # 4. JWT Payload
        payload = {
            "sub": str(user["user_id"]),
            "exp": expire
        }

        # 5. JWT 토큰 생성
        access_token = jwt.encode(
            payload,
            JWT_SECRET_KEY,
            algorithm=JWT_ALGORITHM
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "user": {
                "user_id": user["user_id"],
                "username": user["username"]
            }
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="로그인 중 오류가 발생했습니다."
        )

    finally:
        if conn is not None:
            conn.close()

@app.post("/api/auth/logout")
def logout(user_id: int = Depends(get_current_user_id)):
    return {
        "message": "로그아웃 성공"
    }

@app.get("/api/auth/me")
def get_my_info(
    user_id: int = Depends(get_current_user_id)
):
    conn = None

    try:
        conn = get_connection()

        with conn.cursor(DictCursor) as cursor:
            cursor.execute("""
                SELECT
                    user_id,
                    username,
                    birth,
                    gender,
                    email,
                    created_date
                FROM users
                WHERE user_id = %s
            """, (user_id,))

            user = cursor.fetchone()

        if user is None:
            raise HTTPException(
                status_code=404,
                detail="사용자를 찾을 수 없습니다."
            )

        return user

    finally:
        if conn is not None:
            conn.close()


def verify_websocket_token(token: str):
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM]
        )

        return int(payload["sub"])

    except (jwt.InvalidTokenError, KeyError, ValueError, TypeError):
        return None
