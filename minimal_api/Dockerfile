# 軽量なPythonイメージを使用
FROM python:3.9-slim

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /app

# 依存関係のみをインストール（causalmlは使わない）
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir \
    fastapi==0.95.2 \
    uvicorn==0.22.0 \
    pandas==2.0.1 \
    scikit-learn==1.2.2 \
    numpy==1.24.3

# アプリケーションコードをコピー
COPY minimal_api/ /app/

# モデル保存ディレクトリの作成
RUN mkdir -p /app/saved_models

# ポートの設定
EXPOSE 8000

# 環境変数の設定
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# アプリケーションの実行
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
