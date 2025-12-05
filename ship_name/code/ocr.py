import os
import cv2
from paddleocr import PaddleOCR

class TextRecognizer:
    def __init__(self, input_dir, output_file):
        self.input_dir = input_dir
        self.output_file = output_file
        
        # 初始化 PaddleOCR
        print("Loading PaddleOCR model...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

    def process(self):
        print(f"Start Recognition (Recursive): {self.input_dir}")
        
        results_data = []
        total_success_count = 0  # [新增 1] 初始化計數器
        
        # 改用 os.walk 遞迴搜尋所有子資料夾
        for root, dirs, files in os.walk(self.input_dir):
            # 排序檔案
            valid_files = sorted([f for f in files if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
            
            if not valid_files:
                continue

            # 計算相對路徑 (例如: Taichung)
            rel_path = os.path.relpath(root, self.input_dir)
            
            # 設定圖片存檔的子資料夾路徑，保持結構
            current_save_dir = os.path.join("../result/img_output", rel_path)
            os.makedirs(current_save_dir, exist_ok=True)

            print(f"正在處理資料夾: {rel_path} (共 {len(valid_files)} 張)")

            for f in valid_files:
                path = os.path.join(root, f)
                img = cv2.imread(path)
                if img is None: continue
            
                # 維持您原本的用法
                result = self.ocr.predict_iter(img)
                
                # 解析結果
                for res in result:
                    texts = res["rec_texts"]   # 取出文字
                    scores = res["rec_scores"] # 取出信心度
                    
                    # 簡單過濾置信度低的結果
                    for i in range(len(scores)):
                        if scores[i] > 0.6: 
                            # CSV 紀錄完整的相對路徑
                            full_filename = os.path.join(rel_path, f)
                            print(f"[{full_filename}] 識別結果: {texts[i]} (信心度: {scores[i]:.2f})")
                            results_data.append(f"{full_filename},{texts[i]},{scores[i]:.4f}\n")
                            
                            # [新增 2] 累加成功數量
                            total_success_count += 1
                    
                            # 存檔時傳入對應的子資料夾路徑
                            res.save_to_img(current_save_dir)

        # 儲存結果到 CSV
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as f_out:
            f_out.write("Filename,Detected_Text,Confidence\n")
            f_out.writelines(results_data)
            
        print(f"識別完成，結果已儲存至 {self.output_file}")
        # [新增 3] 顯示總數量
        print(f"本次總共成功辨識出: {total_success_count} 個文字區域")

if __name__ == "__main__":
    recognizer = TextRecognizer(input_dir="../step2_ship_crops", output_file="../result/final_results.csv")
    recognizer.process()