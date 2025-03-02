# conda ベースのイメージを使用
FROM continuumio/miniconda3:latest

# システムパッケージをインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /app

# conda環境の作成
RUN conda create -n causal_env python=3.8 -y

# conda環境を使用するように設定
SHELL ["/bin/bash", "-c"]
ENV PATH=/opt/conda/envs/causal_env/bin:$PATH
RUN echo "source activate causal_env" > ~/.bashrc

# conda-forgeチャネルを追加して科学計算パッケージをインストール（特定バージョンは指定せず）
RUN conda install -n causal_env -y -c conda-forge \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    statsmodels \
    shap

# causalmlをpipでインストール（特定バージョンは指定せず）
RUN /opt/conda/envs/causal_env/bin/pip install causalml

# APIに必要な追加パッケージ
COPY requirements.txt .
RUN /opt/conda/envs/causal_env/bin/pip install -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# モデル保存ディレクトリの作成
RUN mkdir -p /app/saved_models

# ポートの設定
EXPOSE 8000

# 環境変数の設定
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# アプリケーションの実行
CMD ["python", "main.py"]