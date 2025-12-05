import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
from torch.nn.functional import cosine_similarity
import random

class ColorShipReID:
    def __init__(self, yolo_model='../weights/yolo11n.pt', reid_model='osnet_x1_0'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"啟動中... 使用裝置: {self.device}")

        # 載入模型
        self.yolo = YOLO(yolo_model)
        self.extractor = FeatureExtractor(
            model_name=reid_model,
            device=self.device,
            verbose=False
        )

    def generate_color(self, id_seed):
        """
        根據 ID 生成固定且獨特的顏色 (BGR)
        使用隨機種子確保同一個 ID 永遠拿到同一個顏色
        """
        random.seed(id_seed)
        # 避免生成太暗的顏色 (限制範圍 50-255)
        b = random.randint(50, 255)
        g = random.randint(50, 255)
        r = random.randint(50, 255)
        return (b, g, r)

    def get_ships(self, img):
        """偵測並回傳船隻資訊"""
        results = self.yolo(img, verbose=False)
        ships = []
        h, w = img.shape[:2]

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 8: # Class 8 = Boat
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # 邊界限制
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    crop = img[y1:y2, x1:x2]
                    
                    if crop.shape[0] > 10 and crop.shape[1] > 10:
                        ships.append({
                            'box': (x1, y1, x2, y2),
                            'crop': crop,
                            'matched': False,
                            'color': None, # 之後會填入顏色
                            'label': ''
                        })
        return ships

    def run(self, img1_path, img2_path, threshold=0.6):
        # 1. 讀取圖片
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            print("❌ 錯誤：找不到圖片檔案")
            return

        # 2. 影像拼接處理
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        target_h = max(h1, h2)
        
        if h1 < target_h: img1 = cv2.copyMakeBorder(img1, 0, target_h-h1, 0, 0, cv2.BORDER_CONSTANT)
        if h2 < target_h: img2 = cv2.copyMakeBorder(img2, 0, target_h-h2, 0, 0, cv2.BORDER_CONSTANT)
        
        vis_img = np.hstack((img1, img2))
        offset_x = w1 

        # 3. 偵測與特徵提取
        ships1 = self.get_ships(img1)
        ships2 = self.get_ships(img2)
        print(f"偵測數量 -> 左圖: {len(ships1)}, 右圖: {len(ships2)}")

        if ships1 and ships2:
            feats1 = self.extractor([s['crop'] for s in ships1])
            feats2 = self.extractor([s['crop'] for s in ships2])

            matched_indices_2 = set()
            
            # 用一個計數器來產生 Unique ID，確保顏色不重複
            color_id_counter = 0

            # A. 尋找配對 (Matches)
            for i, f1 in enumerate(feats1):
                best_score = -1
                best_idx = -1
                
                for j, f2 in enumerate(feats2):
                    if j in matched_indices_2: continue
                    
                    score = cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
                    if score > best_score:
                        best_score = score
                        best_idx = j
                
                # 若配對成功
                if best_score > threshold:
                    # 1. 產生一個共享顏色
                    shared_color = self.generate_color(color_id_counter)
                    color_id_counter += 1
                    
                    # 2. 左圖船隻設定
                    ships1[i]['matched'] = True
                    ships1[i]['color'] = shared_color
                    ships1[i]['label'] = f"ID:{color_id_counter} ({best_score:.2f})"
                    
                    # 3. 右圖船隻設定
                    ships2[best_idx]['matched'] = True
                    ships2[best_idx]['color'] = shared_color
                    ships2[best_idx]['label'] = f"ID:{color_id_counter}"
                    
                    matched_indices_2.add(best_idx)
                    print(f"配對成功: 左[{i}] == 右[{best_idx}] (顏色ID: {color_id_counter})")

            # B. 處理未配對的 (Unmatched) - 給予各自獨立的顏色
            # 處理左圖剩餘
            for s in ships1:
                if not s['matched']:
                    color_id_counter += 1
                    s['color'] = self.generate_color(color_id_counter)
                    s['label'] = "No Match"
            
            # 處理右圖剩餘
            for j, s in enumerate(ships2):
                if not s['matched']:
                    color_id_counter += 1
                    s['color'] = self.generate_color(color_id_counter)
                    s['label'] = "No Match"

        else:
            # 如果有一邊沒船，所有人都是未配對，給隨機顏色
            color_id = 0
            for s in ships1: 
                s['color'] = self.generate_color(color_id)
                color_id += 1
            for s in ships2: 
                s['color'] = self.generate_color(color_id)
                color_id += 1

        # 4. 繪圖
        # 畫左圖
        for s in ships1:
            x1, y1, x2, y2 = s['box']
            color = s['color']
            # 畫框
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
            # 畫標籤背景 (讓字清楚一點)
            text_size = cv2.getTextSize(s['label'], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(vis_img, (x1, y1 - 25), (x1 + text_size[0], y1), color, -1)
            # 畫標籤文字
            cv2.putText(vis_img, s['label'], (x1, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 畫右圖 (加上 offset)
        for s in ships2:
            x1, y1, x2, y2 = s['box']
            color = s['color']
            # 畫框
            cv2.rectangle(vis_img, (x1 + offset_x, y1), (x2 + offset_x, y2), color, 3)
            # 畫標籤
            text_size = cv2.getTextSize(s['label'], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(vis_img, (x1 + offset_x, y1 - 25), (x1 + offset_x + text_size[0], y1), color, -1)
            cv2.putText(vis_img, s['label'], (x1 + offset_x, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 5. 顯示結果
        # 縮放視窗
        display_h, display_w = vis_img.shape[:2]
        if display_w > 1600:
            scale = 1600 / display_w
            vis_img = cv2.resize(vis_img, None, fx=scale, fy=scale)

        cv2.imshow('Color Match Result', vis_img)
        print("按任意鍵結束...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    image_a = "1.jpeg"
    image_b = "2.jpeg"
    
    app = ColorShipReID()
    app.run(image_a, image_b, threshold=0.6)
