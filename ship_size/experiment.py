import cv2
import numpy as np

# 1. 核心投影類別 (Camera Projector)
# 負責將 2D 影像的像素座標 (Pixel)，透過相機參數轉換為 3D 世界的經緯度 (GPS)。
class CameraProjector:
    def __init__(self, cam_lon, cam_lat, cam_height, hfov_deg, vfov_deg, image_width, image_height, pan_deg, tilt_deg):
        """
        初始化相機參數並計算旋轉矩陣。
        """
        self.cam_loc = np.array([cam_lon, cam_lat, cam_height])
        self.w = image_width
        self.h = image_height
        
        # --- 計算相機內參 ---
        self.fx = (self.w / 2.0) / np.tan(np.radians(hfov_deg) / 2.0) 
        self.fy = (self.h / 2.0) / np.tan(np.radians(vfov_deg) / 2.0) 
        self.cx = self.w / 2.0 
        self.cy = self.h / 2.0
        
        # --- 計算相機外參 ---
        pitch_rad = np.radians(-tilt_deg) # 俯仰角 (向下為正，運算轉負)
        heading_rad = np.radians(pan_deg) # 方位角

        fwd_z = np.sin(pitch_rad)
        fwd_xy_proj = np.cos(pitch_rad)
        fwd_x = fwd_xy_proj * np.sin(heading_rad)
        fwd_y = fwd_xy_proj * np.cos(heading_rad)
        forward = np.array([fwd_x, fwd_y, fwd_z])
        
        right_rad = heading_rad + np.pi/2
        right = np.array([np.sin(right_rad), np.cos(right_rad), 0])
        
        down = np.cross(forward, right)
        
        self.R = np.column_stack((right, down, forward))

    def pixel_to_world(self, u, v):
        """輸入像素座標 (u, v)，回傳投影到海平面 (Height=0) 的經緯度。"""
        x_c = (u - self.cx) / self.fx
        y_c = (v - self.cy) / self.fy
        z_c = 1.0 
        cam_ray = np.array([x_c, y_c, z_c])
        
        world_ray = self.R @ cam_ray
        dx, dy, dz = world_ray 
        
        if dz >= 0: return None # 射線朝上，無交點
        
        t = -self.cam_loc[2] / dz
        
        delta_east = t * dx
        delta_north = t * dy
        
        # 簡易轉換：將公尺偏移量轉換為經緯度偏移
        meters_per_deg_lat = 111132.954 - 559.822 * np.cos(2 * np.radians(self.cam_loc[1]))
        meters_per_deg_lon = 111132.954 * np.cos(np.radians(self.cam_loc[1]))
        
        target_lat = self.cam_loc[1] + (delta_north / meters_per_deg_lat)
        target_lon = self.cam_loc[0] + (delta_east / meters_per_deg_lon)
        
        return target_lon, target_lat

# 2. 距離計算工具 (Haversine Formula)
def calculate_distance(lon1, lat1, lon2, lat2):
    R = 6371000 
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# 3. L型三點選取介面
def get_l_shape_input(image, window_name="Select L-Shape"):
    points = []
    temp_img = image.copy()
    
    print(f"\n--- {window_name} 操作模式 ---")
    print(" 請依序點擊 3 個點:")
    print(" 1. [轉角點 (P1)]：船尾角落 (基準)")
    print(" 2. [長度點 (P2)]：船頭 (定義長度)")
    print(" 3. [寬度點 (P3)]：船側 (定義寬度)")
    print(" [ Q鍵 ] 離開程式")

    def mouse_callback(event, x, y, flags, param):
        nonlocal temp_img
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 3:
                points.append((x, y))
                print(f"-> 點擊位置 #{len(points)}: ({x}, {y})")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        temp_img = image.copy()
        
        # 繪圖
        for i, p in enumerate(points):
            color = (0, 0, 255) # P1, P2 紅
            if i == 2: color = (255, 0, 0) # P3 藍
            
            if i == 0:
                cv2.circle(temp_img, p, 7, (0, 255, 255), -1) 
                cv2.putText(temp_img, "P1 (Corner)", (p[0]+10, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif i == 1:
                cv2.circle(temp_img, p, 5, color, -1)
                cv2.putText(temp_img, "P2", (p[0]+10, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.circle(temp_img, p, 5, color, -1)
                cv2.putText(temp_img, "P3", (p[0]+10, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if len(points) >= 2:
            cv2.arrowedLine(temp_img, points[0], points[1], (0, 255, 0), 2)
        
        if len(points) == 3:
            cv2.arrowedLine(temp_img, points[0], points[2], (255, 255, 0), 2)
            cv2.putText(temp_img, "Press SPACE to Run Experiment", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # 模擬水平框 (HBB) 的樣子，畫出來給使用者看
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            cv2.rectangle(temp_img, (min(xs), min(ys)), (max(xs), max(ys)), (200, 200, 200), 1)

        cv2.imshow(window_name, temp_img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'): return None
        if len(points) == 3 and (key == 32 or key == 13):
            break
            
    cv2.destroyWindow(window_name)
    return points

# 4. 主執行與消融實驗邏輯
def run_ablation_study():
    # --- 參數設定 ---
    BASE_PARAMS = {
        "lon": 120.51722,       
        "lat": 24.29,           
        "height": 37.2,         
        "pan": 273,             
        "tilt": 3.0,            # <--- 基準俯角
        "hfov": 22.1,           
        "vfov": 12.9            
    }
    
    # 請修改為您的圖片路徑
    image_path = 'Taichung_pic/59.jpeg' 
    img = cv2.imread(image_path)
    if img is None:
        print(f"錯誤: 找不到圖片 {image_path}")
        print("請確認路徑或將圖片放在同目錄下。")
        return 

    h, w = img.shape[:2]
    print(f"圖片解析度: {w} x {h}")

    # 取得使用者輸入
    points = get_l_shape_input(img)
    if points is None or len(points) != 3:
        print("使用者取消或未完成 3 點選取。")
        return

    # 定義模擬 HBB (水平框) 的像素點
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    hbb_pixel_1 = (min(xs), max(ys)) # 左下
    hbb_pixel_2 = (max(xs), max(ys)) # 右下

    # 實驗設計：Tilt 敏感度分析
    test_tilts = np.arange(BASE_PARAMS['tilt'] - 0.5, BASE_PARAMS['tilt'] + 0.6, 0.1)
    
    print("\n" + "="*160)
    print(" [消融實驗] 俯角 (Tilt) 變化對 [經緯度] 與 [距離] 的影響")
    print("="*160)
    
    # 格式化標題，使其對齊
    headers = [
        "Tilt", "L-Len", "L-Wid", "HBB-Len", 
        "P1(Corner) GPS", "P2(Head) GPS", "HBB_Left_GPS", "HBB_Right_GPS"
    ]
    print(f"{headers[0]:<7} | {headers[1]:^10} | {headers[2]:^10} | {headers[3]:^10} | {headers[4]:^25} | {headers[5]:^25} | {headers[6]:^25} | {headers[7]:^25}")
    print("-" * 160)

    def fmt_gps(gps):
        """輔助函式：格式化經緯度輸出 (Lon, Lat)"""
        if gps:
            return f"{gps[0]:.6f}, {gps[1]:.6f}"
        return "N/A"

    for t in test_tilts:
        t = round(t, 1) # 避免浮點數誤差
        is_base = (t == BASE_PARAMS['tilt'])
        
        # 建立新的投影器
        proj = CameraProjector(
            BASE_PARAMS["lon"], BASE_PARAMS["lat"], BASE_PARAMS["height"],
            BASE_PARAMS["hfov"], BASE_PARAMS["vfov"],
            w, h, BASE_PARAMS["pan"], tilt_deg=t
        )

        # 1. 計算 L 型 (OBB)
        gps_p1 = proj.pixel_to_world(points[0][0], points[0][1])
        gps_p2 = proj.pixel_to_world(points[1][0], points[1][1])
        gps_p3 = proj.pixel_to_world(points[2][0], points[2][1])

        l_len_str = "N/A"
        l_wid_str = "N/A"
        
        if gps_p1 and gps_p2:
            val = calculate_distance(gps_p1[0], gps_p1[1], gps_p2[0], gps_p2[1])
            l_len_str = f"{val:.2f} m"
            
        if gps_p1 and gps_p3:
            val = calculate_distance(gps_p1[0], gps_p1[1], gps_p3[0], gps_p3[1])
            l_wid_str = f"{val:.2f} m"

        # 計算 水平框 (HBB)
        gps_hbb_1 = proj.pixel_to_world(hbb_pixel_1[0], hbb_pixel_1[1])
        gps_hbb_2 = proj.pixel_to_world(hbb_pixel_2[0], hbb_pixel_2[1])
        
        hbb_len_str = "N/A"
        if gps_hbb_1 and gps_hbb_2:
            val = calculate_distance(gps_hbb_1[0], gps_hbb_1[1], gps_hbb_2[0], gps_hbb_2[1])
            hbb_len_str = f"{val:.2f} m"

        prefix = ">> " if is_base else "   "
        
        # 組合經緯度字串
        p1_str = fmt_gps(gps_p1)
        p2_str = fmt_gps(gps_p2)
        hbb1_str = fmt_gps(gps_hbb_1)
        hbb2_str = fmt_gps(gps_hbb_2)
        
        print(f"{prefix}{t:<7} | {l_len_str:^10} | {l_wid_str:^10} | {hbb_len_str:^10} | {p1_str:^25} | {p2_str:^25} | {hbb1_str:^25} | {hbb2_str:^25}")

    print("-" * 160)
    print("說明：GPS格式為 (經度 Longitude, 緯度 Latitude)。可以觀察數值隨 Tilt 變化的飄移趨勢。")
    print("=" * 160)

    # 顯示結果圖
    output_img = img.copy()
    cv2.arrowedLine(output_img, points[0], points[1], (0, 255, 0), 3)
    cv2.arrowedLine(output_img, points[0], points[2], (255, 255, 0), 3)
    cv2.rectangle(output_img, hbb_pixel_1, (hbb_pixel_2[0], hbb_pixel_1[1]-50), (200, 200, 200), 1)
    cv2.line(output_img, hbb_pixel_1, hbb_pixel_2, (200, 200, 200), 2)
    
    cv2.namedWindow("Experiment Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Experiment Result", output_img)
    print("\n按任意鍵結束程式...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_ablation_study()