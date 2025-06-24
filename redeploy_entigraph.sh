#!/bin/bash

# スクリプトが途中で失敗した場合に、即座に終了させるための設定
set -euo pipefail

# --- 変数 ---
IMAGE_NAME="entigraph_image:latest"
CONTAINER_NAME="entigraph_container"
DOCKERFILE_PATH="." # Dockerfileが含まれるディレクトリへのパス

# --- クリーンアップ：既存コンテナの停止と削除 ---
# `docker ps -q` を使ってコンテナが実行中/存在するかを確認します。
# `|| true` は、コンテナが存在しない場合にスクリプトが終了するのを防ぐために使用します。
echo "Dockerコンテナの停止と削除を試みます: $CONTAINER_NAME..."
docker stop $CONTAINER_NAME || true
docker rm $CONTAINER_NAME || true
echo "コンテナのクリーンアップが完了しました。"

# --- クリーンアップ：既存イメージの削除 ---
# これにより、クリーンな状態からビルドすることが保証されます。
# `|| true` は、イメージが存在しないケースを処理します。
echo "Dockerイメージの削除を試みます: $IMAGE_NAME..."
docker rmi $IMAGE_NAME || true
echo "イメージのクリーンアップが完了しました。"

# --- ビルド：新規イメージの作成 ---
echo "Dockerイメージをビルドします: $IMAGE_NAME (Dockerfileの場所: $DOCKERFILE_PATH)..."
# `--pull` により、セキュリティと一貫性のために最新のベースイメージを使用することが保証されます。
docker build --pull -t $IMAGE_NAME $DOCKERFILE_PATH
# `set -e` がビルドの失敗を自動的に処理するため、手動でのチェックは厳密には不要ですが、
# より明確なエラーメッセージのために残すこともできます。
echo "Dockerイメージ $IMAGE_NAME のビルドが成功しました。"

# --- デプロイ：新規コンテナの実行 ---
echo "コンテナをデプロイします: $CONTAINER_NAME (イメージ: $IMAGE_NAME)..."
# 必要なポートマッピングやボリュームマウントはここに追加してください。
# `--rm` は、コンテナが終了したときに自動的にコンテナを削除します。
# 例: docker run -d --rm -p 8080:80 --name $CONTAINER_NAME $IMAGE_NAME
docker run -d --rm --name $CONTAINER_NAME $IMAGE_NAME
echo "コンテナ $CONTAINER_NAME のデプロイが成功しました。"

# --- 検証 ---
echo "ログを表示する前に数秒待機します..."
sleep 3
echo "--- $CONTAINER_NAME の初期ログ ---"
docker logs --tail 30 $CONTAINER_NAME
echo "--------------------------------------"

echo "$CONTAINER_NAME の再デプロイが完了しました。"
