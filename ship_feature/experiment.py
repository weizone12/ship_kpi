import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
from torch.nn.functional import cosine_similarity
import random
import time
import csv
import os

class ExperimentShipReID:
    def __init__(self, yolo_model='../weights/yolo11n.pt', reid_model='osnet_x1_0', distance_metric='cosine'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.reid_model_name = reid_model
        self.distance_metric = distance_metric # 'cosine' or 'euclidean'
        
        print(f"[{reid_model}] 啟動中... 使用裝置: {self.device}, 算法: {distance_metric}")

        # 載入模型
        self.yolo = YOLO(yolo_model)
        self.extractor = FeatureExtractor(
            model_name=reid_model,
            device=self.device,
            verbose=False
        )
        
        # 實驗數據暫存
        self.logs = []

    def get_ships(self, img):
        """偵測並回傳船隻資訊 (保持原樣)"""
        results = self.yolo(img, verbose=False)
        ships = []
        h, w = img.shape[:2]
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 8:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    crop = img[y1:y2, x1:x2]
                    if crop.shape[0] > 10 and crop.shape[1] > 10:
                        ships.append({'box': (x1, y1, x2, y2), 'crop': crop})
        return ships

    def compute_score(self, feat1, feat2):
        """根據設定計算分數"""
        if self.distance_metric == 'cosine':
            # Cosine: 越大越相似 (-1 ~ 1)
            return cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()
        elif self.distance_metric == 'euclidean':
            # Euclidean: 越小越相似 (0 ~ inf)
            # 為了方便後續邏輯統一，這裡回傳負值，或者需要外部反轉邏輯
            # 這裡我們回傳原始距離
            return torch.dist(feat1, feat2, p=2).item()
        return 0

    def run_experiment(self, img1_path, img2_path, thresholds=[0.6], experiment_name="exp"):
        """
        執行實驗並記錄數據
        thresholds: 可以是一個列表，一次測試多個閾值
        """
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None: return

        # 1. 偵測
        ships1 = self.get_ships(img1)
        ships2 = self.get_ships(img2)
        
        if not ships1 or not ships2: return

        # 2. 特徵提取 (加入計時)
        start_time = time.time()
        feats1 = self.extractor([s['crop'] for s in ships1])
        feats2 = self.extractor([s['crop'] for s in ships2])
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000 # 轉毫秒

        # 計算參數量 (Model Size Proxy)
        param_count = sum(p.numel() for p in self.extractor.model.parameters())

        # 3. 計算所有配對分數矩陣
        all_scores = []
        for i, f1 in enumerate(feats1):
            for j, f2 in enumerate(feats2):
                score = self.compute_score(f1, f2)
                all_scores.append(score)

        # 4. 針對不同的閾值紀錄結果
        for th in thresholds:
            match_count = 0
            
            # 簡單配對邏輯 (Greedy)
            # 注意：這裡簡化了配對邏輯以專注於統計數量
            used_b = set()
            for i, f1 in enumerate(feats1):
                best_s = -999 if self.distance_metric == 'cosine' else 999
                best_j = -1
                
                for j, f2 in enumerate(feats2):
                    if j in used_b: continue
                    s = self.compute_score(f1, f2)
                    
                    if self.distance_metric == 'cosine':
                        if s > best_s: best_s, best_j = s, j
                    else: # euclidean
                        if s < best_s: best_s, best_j = s, j
                
                # 判定閾值
                is_match = False
                if self.distance_metric == 'cosine':
                    if best_s > th: is_match = True
                else: # euclidean
                    if best_s < th: is_match = True
                
                if is_match:
                    match_count += 1
                    used_b.add(best_j)

            # 記錄這一筆實驗數據
            log_entry = {
                'Experiment': experiment_name,
                'Model': self.reid_model_name,
                'Metric': self.distance_metric,
                'Threshold': th,
                'Inference_Time_ms': f"{inference_time:.2f}",
                'Params': param_count,
                'Ships_A': len(ships1),
                'Ships_B': len(ships2),
                'Matches_Found': match_count,
                'Avg_Score': f"{np.mean(all_scores):.4f}" if all_scores else 0
            }
            self.logs.append(log_entry)
            print(f"記錄完畢: {log_entry['Experiment']} | Th:{th} | Matches:{match_count}")

    def save_to_csv(self, filename="ablation_results.csv"):
        if not self.logs: return
        keys = self.logs[0].keys()
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if not file_exists: writer.writeheader()
            writer.writerows(self.logs)
        print(f"數據已儲存至 {filename}")
        self.logs = [] # 清空

if __name__ == "__main__":
    img_a = "1.jpeg"
    img_b = "2.jpeg"
    
    '''實驗一'''
    # OSNet (Baseline)
    exp1_osnet = ExperimentShipReID(reid_model='osnet_x1_0')
    exp1_osnet.run_experiment(img_a, img_b, thresholds=[0.6], experiment_name="Exp1_Model_OSNet")
    exp1_osnet.save_to_csv()
    
    # ResNet50 (Comparison)
    exp1_resnet = ExperimentShipReID(reid_model='resnet50')
    exp1_resnet.run_experiment(img_a, img_b, thresholds=[0.6], experiment_name="Exp1_Model_ResNet")
    exp1_resnet.save_to_csv()

    '''實驗二'''
    # Cosine
    exp2_cos = ExperimentShipReID(reid_model='osnet_x1_0', distance_metric='cosine')
    exp2_cos.run_experiment(img_a, img_b, thresholds=[0.6], experiment_name="Exp2_Metric_Cosine")
    exp2_cos.save_to_csv()

    # Euclidean
    exp2_euc = ExperimentShipReID(reid_model='osnet_x1_0', distance_metric='euclidean')
    exp2_euc.run_experiment(img_a, img_b, thresholds=[20.0], experiment_name="Exp2_Metric_Euclidean")
    exp2_euc.save_to_csv()

    '''實驗三'''
    threshold_list = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 閾值
    exp3 = ExperimentShipReID(reid_model='osnet_x1_0', distance_metric='cosine')
    exp3.run_experiment(img_a, img_b, thresholds=threshold_list, experiment_name="Exp3_Threshold")
    exp3.save_to_csv()