import os
import cv2
import numpy as np
import imagehash
from PIL import Image

class ImagePreprocessor:
    """
        影像前處理類別 (支援遞迴子資料夾版本)
        功能包含：
        1. 亮度檢查 (剔除過曝/過暗)
        2. 去重複 (感知雜湊演算法)
        3. 影像增強 (CLAHE)
    """
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # 參數設定
        self.min_brightness = 40    # 過暗閾值
        self.max_brightness = 220   # 過曝閾值
        self.hash_threshold = 5     # 去重複閾值

        # CLAHE 設定
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  
        
        # 儲存上一張保留圖片的 Hash 值
        self.prev_hash = None

    def apply_clahe_color(self, img):
        """
        使用 CLAHE 增強對比度 (針對彩色圖片的 L 通道處理)
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def process(self):
        print(f"Start Processing: {self.input_dir} -> {self.output_dir}")
        processed_count = 0
        
        # 使用 os.walk 遞迴遍歷所有層級的資料夾
        for root, dirs, files in os.walk(self.input_dir):
            # 排序檔案，確保處理順序 (對去重複很重要)
            valid_files = sorted([f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            
            if not valid_files:
                continue

            # 計算相對路徑，以保持輸出的資料夾結構
            # 例如: input_dir/Taichung -> output_dir/Taichung
            rel_path = os.path.relpath(root, self.input_dir)
            current_output_dir = os.path.join(self.output_dir, rel_path)
            os.makedirs(current_output_dir, exist_ok=True)
            
            # 重要：切換資料夾時，重置上一張圖片的 Hash
            # 避免「資料夾A的最後一張」跟「資料夾B的第一張」被誤判為重複
            self.prev_hash = None
            
            print(f"正在處理資料夾: {rel_path} (共 {len(valid_files)} 張)")

            for f in valid_files:
                input_path = os.path.join(root, f)
                img = cv2.imread(input_path)
                if img is None: continue
                
                # 1. 亮度檢查
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mean_bright = np.mean(gray)
                if mean_bright < self.min_brightness or mean_bright > self.max_brightness:
                    print(f"  [剔除-亮度異常] {f}: {mean_bright:.1f}")
                    continue
                    
                # 2. 去重複檢查
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                curr_hash = imagehash.phash(pil_img)
                
                if self.prev_hash and (curr_hash - self.prev_hash < self.hash_threshold):
                    print(f"  [剔除-重複影像] {f}")
                    continue
                self.prev_hash = curr_hash
                
                # 3. 影像增強 (CLAHE)
                enhanced_img = self.apply_clahe_color(img)
                
                # 儲存結果
                output_path = os.path.join(current_output_dir, f)
                cv2.imwrite(output_path, enhanced_img)
                processed_count += 1
            
        print(f"全部處理完成，共保留 {processed_count} 張圖片。")

if __name__ == "__main__":
    # 請根據您的實際路徑修改這裡
    # 假設您的 img 資料夾在上一層
    processor = ImagePreprocessor(input_dir="../img", output_dir="../step1_cleaned")
    processor.process()