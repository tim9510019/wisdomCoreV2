import os
from huggingface_hub import HfApi

# ==========================================
# [ 上傳配置區 ]
# ==========================================
# 1. 目標 Repo 名稱 (格式：您的帳號名稱/專案名稱)
REPO_ID = "tim9510019/AGIV2-1300M-blackwell-CPT" 

# 2. 欲上傳的目標路徑
MODEL_PATH = "./agiv2_cpt_checkpoints_1KST/best_cpt_model.pth"

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

    # 執行上傳邏輯
    if os.path.isfile(MODEL_PATH):
        print(f"📦 偵測到單一檔案，準備上傳: {MODEL_PATH} ...")
        api.upload_file(
            path_or_fileobj=MODEL_PATH,
            path_in_repo=os.path.basename(MODEL_PATH),
            repo_id=REPO_ID,
            repo_type="model"
        )
    elif os.path.isdir(MODEL_PATH):
        print(f"📁 偵測到資料夾，準備批次上傳: {MODEL_PATH} ...")
        api.upload_folder(
            folder_path=MODEL_PATH,
            repo_id=REPO_ID,
            repo_type="model"
        )
    else:
        print(f"❌ 找不到指定的路徑 {MODEL_PATH}，請確認訓練檔案是否存在。")
        return

    print(f"\n🎉 上傳完成！您的模型已成功推送至雲端：https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()