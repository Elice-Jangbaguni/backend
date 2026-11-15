import os
import pymysql
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        charset="utf8mb4"
    )


if __name__ == "__main__":
    conn = get_connection()

    with conn.cursor() as cursor:
        cursor.execute("SELECT DATABASE()")
        print("연결된 DB:", cursor.fetchone())

        cursor.execute("SELECT user_id, username FROM users")
        print("사용자 목록:", cursor.fetchall())

    conn.close()