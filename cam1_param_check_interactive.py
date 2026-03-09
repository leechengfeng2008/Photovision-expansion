from __future__ import annotations


def ask_yn(prompt: str) -> bool:
    while True:
        ans = input(prompt + " [y/n]: ").strip().lower()
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("請輸入 y 或 n")


def main() -> None:
    print("=== CAM1 參數確認互動檢查 ===")
    print("建議測試前提：")
    print("1. 先只開 Camera1")
    print("2. 球都放在地面")
    print("3. YAW 測試時盡量保持球與相機距離差不多")
    print("4. PITCH / 距離測試時把球放在畫面水平中心附近")
    print()

    notes: list[str] = []
    recs: list[str] = []

    print("--- A. YAW 方向檢查 ---")
    center_zero = ask_yn("1) 把球放在 Camera1 畫面正中央時，yaw 是否接近 0？")
    left_increase = ask_yn("2) 把球往畫面左邊移動時，yaw 數值是否變大？")
    right_decrease = ask_yn("3) 把球往畫面右邊移動時，yaw 數值是否變小？")

    if center_zero:
        notes.append("YAW 中心點大致正常：球在畫面中央時 yaw 接近 0。")
    else:
        notes.append("YAW 中心點有偏移：球在畫面中央時 yaw 不接近 0。")
        recs.append("先不要急著改 YAW_SIGN；先檢查 PhotonVision crosshair / pipeline 中心、鏡頭安裝角度、或是否需要記錄一個固定 yaw bias。")

    if left_increase and right_decrease:
        notes.append("CAM1 原始 yaw 慣例看起來是：左正右負。")
        recs.append("若你沿用目前公式 bearing_robot_deg = camera_yaw_offset_deg - YAW_SIGN * yaw_deg，建議先試 YAW_SIGN = -1.0。")
    elif (not left_increase) and (not right_decrease):
        notes.append("CAM1 原始 yaw 慣例看起來是：右正左負。")
        recs.append("若你沿用目前公式 bearing_robot_deg = camera_yaw_offset_deg - YAW_SIGN * yaw_deg，建議先試 YAW_SIGN = +1.0。")
    else:
        notes.append("YAW 方向結果不一致。")
        recs.append("代表 yaw 方向測試不夠穩，或球移動時距離/遮擋也變了。請重新做『左、右、同距離』測試。")

    print()
    print("--- B. PITCH 方向檢查 ---")
    far_pitch_increase = ask_yn("4) 球在地面上沿著前方移遠時，pitch 數值是否變大？")
    dist_monotonic = ask_yn("5) 用你現在的 distance_calculate()，球移遠時估計 distance 是否也跟著變大？")

    if far_pitch_increase:
        notes.append("PITCH 觀察：球越遠，pitch 越大。")
        recs.append("這通常和你現在的公式 total_deg = camera_pitch_deg - pitch_deg 相容。")
    else:
        notes.append("PITCH 觀察：球越遠，pitch 沒有變大，可能是變小。")
        recs.append("先優先測試把 distance_utils.py 裡的 total_deg = camera_pitch_deg - pitch_deg 改成 total_deg = camera_pitch_deg + pitch_deg。")

    if dist_monotonic:
        notes.append("距離單調性正常：球越遠，估計 distance 也越大。")
    else:
        notes.append("距離單調性異常：球越遠，估計 distance 沒有跟著變大。")
        recs.append("若第 4 題也是 n，優先改 pitch 公式正負號。若第 4 題是 y 但第 5 題仍是 n，請檢查 CAMERA_PITCH_DEG 的正負與大小。")

    print()
    print("--- C. 距離量級檢查 ---")
    dist_too_large = ask_yn("6) 在多個距離點上，估計 distance 是否普遍偏大？")
    dist_too_small = ask_yn("7) 在多個距離點上，估計 distance 是否普遍偏小？")

    if dist_too_large and dist_too_small:
        notes.append("距離誤差不是單一方向，可能是公式方向對但模型/量測不穩。")
        recs.append("先固定 3~4 個實際距離點做表格，比較每點誤差；這通常表示不是單純一個常數能修好。")
    elif dist_too_large:
        notes.append("估計距離普遍偏大。")
        recs.append("優先檢查：CAMERA_PITCH_DEG 是否設太小、CAMERA_HEIGHT_M 是否設太高、TARGET_HEIGHT_M 是否設太低。")
    elif dist_too_small:
        notes.append("估計距離普遍偏小。")
        recs.append("優先檢查：CAMERA_PITCH_DEG 是否設太大、CAMERA_HEIGHT_M 是否設太低、TARGET_HEIGHT_M 是否設太高。")
    else:
        notes.append("估計距離量級大致合理。")

    print()
    print("=== 統整結果 ===")
    for line in notes:
        print("-", line)

    print()
    print("=== 建議修改方向 ===")
    if recs:
        seen = set()
        for line in recs:
            if line not in seen:
                print("-", line)
                seen.add(line)
    else:
        print("- 目前看起來 YAW / PITCH / 距離方向都大致合理，可開始進入數值微調。")

    print()
    print("=== 建議你實際量的 4 個距離點 ===")
    print("- 0.60 m")
    print("- 1.00 m")
    print("- 1.50 m")
    print("- 2.00 m")
    print("每個點至少記：yaw、pitch、estimated distance、actual distance。")


if __name__ == "__main__":
    main()
