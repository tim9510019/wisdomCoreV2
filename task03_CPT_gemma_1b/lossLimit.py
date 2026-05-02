import numpy as np
from scipy.optimize import curve_fit

# 載入觀測到的實體數據
D = np.array([32000000, 64000000, 96000000, 128000000, 160000000, 192000000, 224000000])
L = np.array([6.020431, 5.146315, 5.07834, 4.93737, 4.751762, 4.616858, 4.315414])

# 定義降維後的 Scaling Law 函數
def scaling_law(D, E_prime, B, beta):
    return E_prime + B / (D ** beta)

# 拓撲觀測：設定初始猜測值 (經驗法則中 beta 常落於 0.1~0.5 之間)
initial_guess = [3.0, 1e5, 0.3]

try:
    # 進行曲線擬合
    popt, pcov = curve_fit(scaling_law, D, L, p0=initial_guess, maxfev=20000)
    E_prime, B, beta = popt
    
    print("=== 量子態塌縮：擬合結果 ===")
    print(f"局部極限 E' (E + A/N^alpha) = {E_prime:.6f}")
    print(f"常數 B = {B:.6e}")
    print(f"數據縮放指數 beta = {beta:.6f}")
    
except RuntimeError as e:
    print(f"無法收斂，需要調整初始猜測值或收集更多數據點: {e}")