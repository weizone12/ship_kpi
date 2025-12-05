import cv2
import numpy as np

# 1. 核心投影類別 (Camera Projector)
# 負責將 2D 影像的像素座標 (Pixel)，透過相機參數轉換為 3D 世界的經緯度 (GPS)。
class CameraProjector:
    def __init__(self, cam_lon, cam_lat, cam_height, hfov_deg, vfov_deg, image_width, image_height, pan_deg, tilt_deg):
        """
        初始化相機參數並計算旋轉矩陣。
        :param cam_lon: 相機經度
        :param cam_lat: 相機緯度
        :param cam_height: 相機架設高度 (公尺)
        :param hfov_deg: 水平視角 (Horizontal FOV)
        :param vfov_deg: 垂直視角 (Vertical FOV)
        :param image_width: 影像寬度
        :param image_height: 影像高度
        :param pan_deg: 方位角 (0=北, 90=東)
        :param tilt_deg: 俯角 (向下看為正值)
        """
        # 儲存相機位置向量 [經度, 緯度, 高度]
        self.cam_loc = np.array([cam_lon, cam_lat, cam_height])
        self.w = image_width
        self.h = image_height
        
        # 計算相機內參 (Intrinsics)
        # 利用 FOV 計算虛擬焦距 (Focal Length)，這是像素與角度轉換的比例尺
        # fx, fy 代表成像平面到光心的像素距離
        self.fx = (self.w / 2.0) / np.tan(np.radians(hfov_deg) / 2.0) 
        self.fy = (self.h / 2.0) / np.tan(np.radians(vfov_deg) / 2.0) 
        # cx, cy 是影像的光學中心 (通常是影像解析度的一半)
        self.cx = self.w / 2.0 
        self.cy = self.h / 2.0
        
        # 計算相機外參 (Extrinsics - 旋轉矩陣)
        # 將輸入的角度轉換為弧度 (Radians)
        pitch_rad = np.radians(-tilt_deg) # 俯仰角 (因數學定義抬頭為正，低頭為負，故加負號)
        heading_rad = np.radians(pan_deg) # 方位角

        # 1. 計算「前方 (Forward)」向量：鏡頭看向哪裡
        # 利用球座標公式將 Pan/Tilt 轉為 XYZ 向量
        fwd_z = np.sin(pitch_rad)                 # Z分量 (高度變化)
        fwd_xy_proj = np.cos(pitch_rad)           # 水平投影長度
        fwd_x = fwd_xy_proj * np.sin(heading_rad) # X分量 (東向)
        fwd_y = fwd_xy_proj * np.cos(heading_rad) # Y分量 (北向)
        forward = np.array([fwd_x, fwd_y, fwd_z])
        
        # 2. 計算「右方 (Right)」向量
        # 假設相機無側滾 (Roll)，右邊就是方位角 + 90度，且保持水平
        right_rad = heading_rad + np.pi/2
        right = np.array([np.sin(right_rad), np.cos(right_rad), 0])
        
        # 3. 計算「下方 (Down)」向量
        # 利用向量外積 (Cross Product): 前 x 右 = 下
        down = np.cross(forward, right)
        
        # 4. 組合旋轉矩陣 R (Column Stack)
        # 這個矩陣能將「相機座標系」轉譯為「世界座標系 (ENU)」
        self.R = np.column_stack((right, down, forward))

    def pixel_to_world(self, u, v):
        """
        輸入像素座標 (u, v)，回傳投影到海平面 (Height=0) 的經緯度。
        """
        # 1. 將像素座標轉為相機座標系的歸一化方向向量
        # (u - cx) / fx 去除焦距影響
        x_c = (u - self.cx) / self.fx
        y_c = (v - self.cy) / self.fy
        z_c = 1.0 # 假設成像平面在前方 1 單位處
        cam_ray = np.array([x_c, y_c, z_c])
        
        # 2. 旋轉向量：相機座標 -> 世界座標
        world_ray = self.R @ cam_ray
        dx, dy, dz = world_ray # 得到射線在世界座標的三軸分量
        
        # 3. 檢查射線是否朝上 (dz >= 0)
        # 如果射線往上或水平，永遠碰不到海面，回傳 None
        if dz >= 0: return None 
        
        # 4. 射線延伸求交點 (Ray Casting)
        # 計算射線要延伸多長 (t) 才會碰到海平面 (Z=0)
        # 公式: 相機高度 + t * 垂直分量 = 0  =>  t = -高度 / 垂直分量
        t = -self.cam_loc[2] / dz
        
        # 計算水平位移 (公尺)
        delta_east = t * dx
        delta_north = t * dy
        
        # 5. 公尺轉經緯度 (Geodetic Conversion)
        # 計算該緯度下，每度經度與緯度代表多少公尺
        meters_per_deg_lat = 111132.954 - 559.822 * np.cos(2 * np.radians(self.cam_loc[1]))
        meters_per_deg_lon = 111132.954 * np.cos(np.radians(self.cam_loc[1]))
        
        # 加上位移得到目標經緯度
        target_lat = self.cam_loc[1] + (delta_north / meters_per_deg_lat)
        target_lon = self.cam_loc[0] + (delta_east / meters_per_deg_lon)
        
        return target_lon, target_lat

# 2. 距離計算工具 (Haversine Formula)
def calculate_distance(lon1, lat1, lon2, lat2):
    """
    計算地球表面兩點 (GPS) 間的直線距離 (單位：公尺)。
    """
    R = 6371000 # 地球平均半徑 (公尺)
    
    # 角度轉弧度
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    # Haversine 公式
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

# 3. L型三點選取介面 (UI Interaction)
def get_l_shape_input(image, window_name="Select L-Shape"):
    """
    開啟視窗讓使用者點擊 3 個點。
    已移除 R 鍵重置功能。
    """
    points = []
    temp_img = image.copy()
    
    print(f"\n--- {window_name} 操作模式 (L型) ---")
    print(" 請依序點擊 3 個點:")
    print(" 1. [轉角點 (P1)]：通常是船尾角落 (或船頭) - 作為基準點")
    print(" 2. [長度點 (P2)]：船的另一端 - 定義船長")
    print(" 3. [寬度點 (P3)]：船的另一側 - 定義船寬")
    print(" [ Q鍵 ] 離開程式")

    # 滑鼠回調函式 (Mouse Callback)
    def mouse_callback(event, x, y, flags, param):
        nonlocal temp_img
        # 監聽滑鼠左鍵點擊
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 3: # 限制只能點 3 點
                points.append((x, y))
                print(f"-> 點擊位置 #{len(points)}: ({x}, {y})")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        temp_img = image.copy()
        
        # 繪圖邏輯 (即時顯示點和線)
        for i, p in enumerate(points):
            color = (0, 0, 255) # P1, P2 用紅色
            if i == 2: color = (255, 0, 0) # P3 用藍色
            
            # 特殊標註 P1 是轉角
            if i == 0:
                cv2.circle(temp_img, p, 7, (0, 255, 255), -1) # 黃點 P1
                cv2.putText(temp_img, "Corner", (p[0]+10, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.circle(temp_img, p, 5, color, -1)

        # 畫箭頭連線
        if len(points) >= 2:
            # P1 -> P2 (長度軸)
            cv2.arrowedLine(temp_img, points[0], points[1], (0, 255, 0), 2)
        
        if len(points) == 3:
            # P1 -> P3 (寬度軸)
            cv2.arrowedLine(temp_img, points[0], points[2], (255, 255, 0), 2)
            cv2.putText(temp_img, "Press SPACE to Calculate", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow(window_name, temp_img)
        key = cv2.waitKey(1) & 0xFF
        
        # Q 鍵離開
        if key == ord('q'): return None
        
        # 收集滿 3 點後，按 Space(32) 或 Enter(13) 結束選取
        if len(points) == 3 and (key == 32 or key == 13):
            break
            
    cv2.destroyWindow(window_name)
    return points

# 4. 主執行
def run_manual_measure():
    # 攝影機參數設定
    CAM_PARAMS = {
        "lon": 120.51722,       # 經度
        "lat": 24.29,           # 緯度
        "height": 37.2,         # 高度 (m)
        "pan": 273,             # 方位角
        "tilt": 3.0,            # 俯角 (向下看 3 度)
        "hfov": 22.1,           # 水平視角
        "vfov": 12.9            # 垂直視角
    }
    
    image_path = 'Taichung pic/63.jpeg' 
    img = cv2.imread(image_path)
    
    # 檢查圖片是否讀取成功
    if img is None:
        print(f"錯誤: 找不到圖片 {image_path}")
        return 

    h, w = img.shape[:2]
    print(f"圖片解析度: {w} x {h}")

    # 初始化投影物件
    projector = CameraProjector(
        CAM_PARAMS["lon"], CAM_PARAMS["lat"], CAM_PARAMS["height"],
        CAM_PARAMS["hfov"], CAM_PARAMS["vfov"],
        w, h,
        CAM_PARAMS["pan"], CAM_PARAMS["tilt"]
    )

    # 取得使用者輸入的 3 個點
    points = get_l_shape_input(img)
    
    if points is None or len(points) != 3:
        print("使用者取消或未完成 3 點選取。")
        return

    # 將像素座標轉換為經緯度
    gps_points = []
    
    print("\n" + "="*40)
    print("       點位經緯度資訊 (GPS Coordinates)       ")
    print("="*40)
    
    point_names = ["P1 (轉角點)", "P2 (長度點)", "P3 (寬度點)"]
    
    for i, (px, py) in enumerate(points):
        # 呼叫轉換函式
        coord = projector.pixel_to_world(px, py)
        
        if coord:
            gps_points.append(coord)
            # 列印經緯度 (coord[0]=Lon, coord[1]=Lat)
            print(f" {point_names[i]} [Pixel: {px:4},{py:4}] -> 經度: {coord[0]:.8f}, 緯度: {coord[1]:.8f}")
        else:
            print(f" {point_names[i]} [Pixel: {px:4},{py:4}] -> 指向天空 (無法計算)")
            # 若有任一點無效，則終止程式
            return

    # 計算 L 型長寬
    # 長度 = P1 到 P2 的距離
    length = calculate_distance(gps_points[0][0], gps_points[0][1], 
                                gps_points[1][0], gps_points[1][1])

    # 寬度 = P1 到 P3 的距離
    width = calculate_distance(gps_points[0][0], gps_points[0][1], 
                               gps_points[2][0], gps_points[2][1])

    print("-" * 40)
    print(f" 測量結果 (Measurement Results):")
    print(f"  > 推算船長 (L): {length:.2f} 公尺")
    print(f"  > 推算船寬 (W): {width:.2f} 公尺")
    print("=" * 40)

    # 繪製結果圖
    output_img = img.copy()
    
    # 畫線: 長度(綠色), 寬度(黃色)
    cv2.arrowedLine(output_img, points[0], points[1], (0, 255, 0), 3)   
    cv2.arrowedLine(output_img, points[0], points[2], (255, 255, 0), 3) 
    
    # 畫點: P1(黃), P2(紅), P3(藍)
    cv2.circle(output_img, points[0], 8, (0, 255, 255), -1) 
    cv2.circle(output_img, points[1], 6, (0, 0, 255), -1)   
    cv2.circle(output_img, points[2], 6, (255, 0, 0), -1)   
    
    # 計算文字顯示位置 (放在線段中間)
    mid_L_x = (points[0][0] + points[1][0]) // 2
    mid_L_y = (points[0][1] + points[1][1]) // 2
    
    mid_W_x = (points[0][0] + points[2][0]) // 2
    mid_W_y = (points[0][1] + points[2][1]) // 2
    
    # 標示長寬數值
    cv2.putText(output_img, f"L: {length:.1f}m", (mid_L_x, mid_L_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(output_img, f"W: {width:.1f}m", (mid_W_x, mid_W_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # 顯示最終視窗
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", output_img)
    print("\n按任意鍵結束程式...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 程式進入點
if __name__ == "__main__":
    run_manual_measure()
