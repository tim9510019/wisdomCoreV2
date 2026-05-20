import os
from huggingface_hub import HfApi

# ==========================================
# [ 上傳配置區 ]
# ==========================================
# 1. 目標 Repo 名稱 (格式：您的帳號名稱/專案名稱)
REPO_ID = "tim9510019/AGIV2-1300M-blackwell-CPT_GT"

# 2. 欲上傳的目標路徑 (模型檔案)
MODEL_PATH = "./agiv2_cpt_checkpoints_1KSGT/best_cpt_model.pth"

# 3. 欲上傳的 Log 檔案路徑
LOG_PATH = "agiv2_cpt_1kSGT_log.csv"


def main():
    print(f"🚀 啟動 Hugging Face 上傳程序 (將自動使用您剛才登入的憑證)...")

    # 直接初始化 API，它會自動抓取 CLI 登入的 Token
    api = HfApi()

    try:
        api.create_repo(repo_id=REPO_ID, exist_ok=True, repo_type="model")
        print(f"✅ 目標 Repo 確認完畢: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"❌ 建立/讀取 Repo 失敗，請確認您的帳號名稱與權限: {e}")
        return

    # --- 執行模型上傳邏輯 ---
    if os.path.isfile(MODEL_PATH):
        print(f"📦 偵測到模型檔案，準備上傳: {MODEL_PATH} ...")
        api.upload_file(
            path_or_fileobj=MODEL_PATH,
            path_in_repo=os.path.basename(MODEL_PATH),
            repo_id=REPO_ID,
            repo_type="model",
        )
    elif os.path.isdir(MODEL_PATH):
        print(f"📁 偵測到資料夾，準備批次上傳: {MODEL_PATH} ...")
        api.upload_folder(folder_path=MODEL_PATH, repo_id=REPO_ID, repo_type="model")
    else:
        print(f"⚠️ 找不到指定的模型路徑 {MODEL_PATH}，略過此項。")

    # --- 執行 CSV Log 上傳邏輯 ---
    if os.path.isfile(LOG_PATH):
        print(f"📊 偵測到 Log 檔案，準備上傳: {LOG_PATH} ...")
        api.upload_file(
            path_or_fileobj=LOG_PATH,
            path_in_repo=os.path.basename(LOG_PATH),
            repo_id=REPO_ID,
            repo_type="model",
        )
    else:
        print(f"❌ 找不到指定的 CSV 檔案 {LOG_PATH}，請確認檔案是否存在。")

    print(f"\n🎉 所有上傳動作已處理完成！")
    print(f"🔗 查看您的模型與 Log：https://huggingface.co/{REPO_ID}/tree/main")


if __name__ == "__main__":
    main()
