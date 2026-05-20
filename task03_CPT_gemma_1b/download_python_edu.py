"""
download_smollm_s3.py — SmolLM Corpus S3 本機化下載腳本
======================================================
將 HuggingFaceTB/smollm-corpus (python-edu) 中僅具備 blob_id 的 Metadata，
透過多執行緒去 S3 實際下載原始碼，並寫入本機的 Parquet 檔案。
這能確保後續訓練腳本 100% 穩定，不會受到網路波動影響。
"""

import os
import sys
import time
import concurrent.futures
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm
import boto3
import botocore
from smart_open import open as smart_open

# ==========================================
# [ 全局配置 ]
# ==========================================
DATASET_PATH = "HuggingFaceTB/smollm-corpus"
DATASET_NAME = "python-edu"
SPLIT = "train"
OUTPUT_FILE = "./python_edu_local.parquet"
MAX_WORKERS = 32
WRITE_BATCH_SIZE = 10000
MAX_RETRIES = 10

# S3 無簽名存取配置
s3_client = boto3.client(
    "s3",
    region_name="us-west-2",
    config=botocore.config.Config(signature_version=botocore.UNSIGNED),
)


def fetch_blob(blob_id: str) -> str:
    """具備阻塞重試機制的 S3 抓取函數"""
    s3_url = f"s3://softwareheritage/content/{blob_id}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with smart_open(
                s3_url, "rb", compression=".gz", transport_params={"client": s3_client}
            ) as f:
                return f.read().decode("utf-8", errors="ignore")
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"\n🚨 [放棄] S3 區塊 {blob_id} 徹底失效 ({e})")
                return ""
            time.sleep(1.5)
    return ""


def main():
    print("🚀 啟動 SmolLM-Corpus (python-edu) S3 本機化下載引擎...")

    # 讀取 Metadata (不使用 streaming 以取得總數，Metadata 佔用記憶體不大)
    print("📥 正在載入 HuggingFace Metadata (僅讀取清單)...")
    dataset = load_dataset(DATASET_PATH, name=DATASET_NAME, split=SPLIT)
    total_records = len(dataset)
    print(f"📊 總計發現 {total_records:,} 筆 Python 檔案紀錄。")

    schema = pa.schema([("blob_id", pa.string()), ("text", pa.string())])

    # 若檔案已存在，防呆警告
    if os.path.exists(OUTPUT_FILE):
        print(f"⚠️ 警告：{OUTPUT_FILE} 已存在！為了避免資料覆蓋，請先手動刪除或備份。")
        sys.exit(1)

    batch_data = []
    success_count = 0
    failed_count = 0

    print(f"⚡ 啟動並行下載 (最大執行緒: {MAX_WORKERS})...")
    with pq.ParquetWriter(OUTPUT_FILE, schema) as writer:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            with tqdm(
                total=total_records, desc="Downloading S3 Blobs", smoothing=0.1
            ) as pbar:

                # 為了避免將幾百萬個 Future 塞爆記憶體，我們動態維持一個 pending set
                pending = set()
                iterator = iter(dataset)

                while True:
                    # 補滿 pending 任務池到 2 倍的 Worker 數量
                    while len(pending) < MAX_WORKERS * 2:
                        try:
                            row = next(iterator)
                            blob_id = row.get("blob_id")
                            if blob_id:
                                future = executor.submit(
                                    lambda b=blob_id: (b, fetch_blob(b))
                                )
                                pending.add(future)
                            else:
                                pbar.update(1)
                                failed_count += 1
                        except StopIteration:
                            break

                    if not pending:
                        break

                    # 等待「至少一個」下載完成再繼續
                    done, pending = concurrent.futures.wait(
                        pending, return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    for future in done:
                        pbar.update(1)
                        b_id, text = future.result()

                        if text:
                            batch_data.append({"blob_id": b_id, "text": text})
                            success_count += 1
                        else:
                            failed_count += 1

                        # 達到 Batch Size 寫入硬碟並清空記憶體
                        if len(batch_data) >= WRITE_BATCH_SIZE:
                            table = pa.Table.from_pylist(batch_data, schema=schema)
                            writer.write_table(table)
                            batch_data = []

            # 寫入最後殘餘的 batch
            if batch_data:
                table = pa.Table.from_pylist(batch_data, schema=schema)
                writer.write_table(table)

    print("\n✅ 下載與轉換完成！")
    print(f"📁 檔案已儲存至：{OUTPUT_FILE}")
    print(f"📈 成功下載寫入：{success_count:,} 筆")
    print(f"📉 徹底失效放棄：{failed_count:,} 筆")

    # ==========================================
    # [ 雲端同步 ] 自動推送到 HuggingFace Hub
    # ==========================================
    print("\n🚀 準備將本機 Parquet 同步至 HuggingFace Hub (tim9510019/python-edu)...")
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        repo_id = "tim9510019/python-edu"

        # 確保 Dataset Repository 存在
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

        print(f"⏳ 正在上傳 {OUTPUT_FILE} (檔案可能高達數 GB，請耐心等候)...")
        api.upload_file(
            path_or_fileobj=OUTPUT_FILE,
            path_in_repo="python_edu_local.parquet",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Initial commit: S3 extracted smollm python-edu corpus",
        )
        print(f"🎉 上傳大功告成！")
        print(f"🔗 你的私有/公開資料集網址：https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"\n❌ HuggingFace 上傳失敗：{e}")
        print("不用擔心，本地端的檔案依然安全，你可以稍後手動上傳。")


if __name__ == "__main__":
    main()
