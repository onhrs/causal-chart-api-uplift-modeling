import uvicorn
import os
from app.api import app

if __name__ == "__main__":
    # 環境変数からポートを取得（デフォルトは8000）
    port = int(os.environ.get("PORT", 8000))
    
    # アプリケーションの起動
    uvicorn.run("app.api:app", host="0.0.0.0", port=port, reload=True)
