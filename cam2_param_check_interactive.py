from __future__ import annotations

import ast
import math
import os
import re
import statistics
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 使用 project 的 distance model；若找不到則用內建版本。
# ---------------------------------------------------------------------------
try:
    from utils.distance_utils import distance_calculate as project_distance_calculate
except Exception:
    project_distance_calculate = None


@dataclass
class TestPoint:
    label: str
    actual_m: float
    raw_pitch_deg: float
    raw_yaw_deg: Optional[float] = None
    raw_area: Optional[float] = None


@dataclass
class EvalResult:
    mode: str
    est_list: List[Optional[float]]
    valid_pairs: List[Tuple[float, float]]
    mean_error: Optional[float]
    mean_abs_error: Optional[float]
    std_error: Optional[float]
    slope_err_vs_actual: Optional[float]
    intercept_err_vs_actual: Optional[float]
    corr_err_vs_actual: Optional[float]


@dataclass
class FitResult:
    mode: str
    camera_pitch_deg: float
    camera_height_m: float
    delta_height_m: float
    mae: float
    rmse: float
    mean_error: float
    est_list: List[Optional[float]]
    num_used: int


@dataclass
class YawAnalysis:
    raw_convention: Optional[str]
    recommended_yaw_sign: Optional[float]
    center_bias_deg: Optional[float]
    offset_delta_deg: Optional[float]
    notes: List[str]
    recommendations: List[str]


# Camera2 預設值（鏡像安裝，yaw offset 為負）
MAIN_DEFAULTS = {
    "CAMERA_HEIGHT_M":   0.527,
    "CAMERA_PITCH_DEG":  21.0,
    "TARGET_HEIGHT_M":   0.075,
    "CAMERA2_YAW_OFFSET_DEG": -30.0,
    "CAMERA2_YAW_SIGN":  -1.0,
}


def ask_str(prompt: str, default: Optional[str] = None) -> str:
    while True:
        if default is None:
            s = input(f"{prompt}: ").strip()
        else:
            s = input(f"{prompt} [{default}]: ").strip()
            if s == "":
                return default
        if s != "":
            return s


def ask_float(prompt: str, default: Optional[float] = None, allow_blank: bool = False) -> Optional[float]:
    while True:
        shown = None if default is None else f"{default:.6g}"
        if shown is None:
            s = input(f"{prompt}: ").strip()
        else:
            s = input(f"{prompt} [{shown}]: ").strip()
            if s == "":
                return default
        if s == "" and allow_blank:
            return None
        try:
            return float(s)
        except ValueError:
            print("請輸入數字；若要跳過，直接 Enter。" if allow_blank else "請輸入數字。")


def ask_yes_no(prompt: str, default: Optional[bool] = None) -> bool:
    suffix = " [y/n]"
    if default is True:
        suffix = " [Y/n]"
    elif default is False:
        suffix = " [y/N]"
    while True:
        s = input(prompt + suffix + ": ").strip().lower()
        if s == "" and default is not None:
            return default
        if s in {"y", "yes"}:
            return True
        if s in {"n", "no"}:
            return False
        print("請輸入 y 或 n。")


def parse_numeric_literal(text: str) -> Optional[float]:
    try:
        value = ast.literal_eval(text)
    except Exception:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def load_main_defaults(main_py_path: str) -> dict:
    values = dict(MAIN_DEFAULTS)
    if not os.path.exists(main_py_path):
        return values
    pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^#\n]+)", re.MULTILINE)
    try:
        text = open(main_py_path, "r", encoding="utf-8").read()
    except Exception:
        return values
    found = {}
    for name, expr in pattern.findall(text):
        if name in values:
            parsed = parse_numeric_literal(expr.strip())
            if parsed is not None:
                found[name] = parsed
    values.update(found)
    return values


def internal_distance_calculate(
    pitch_deg_list: Iterable[float],
    camera_height_m: float,
    camera_pitch_deg: float,
    target_height_m: float,
    mode: str = "minus",
) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    delta_height = camera_height_m - target_height_m
    for pitch_deg in pitch_deg_list:
        pitch_deg = float(pitch_deg)
        total_deg = (camera_pitch_deg - pitch_deg) if mode == "minus" else (camera_pitch_deg + pitch_deg)
        total_rad = math.radians(total_deg)
        tanvalue = math.tan(total_rad)
        if abs(tanvalue) < 1e-9:
            out.append(None)
            continue
        distance = delta_height / tanvalue
        out.append(distance if distance > 0 else None)
    return out


def calculate_distances(pitch_deg_list, camera_height_m, camera_pitch_deg, target_height_m, mode="minus"):
    pitch_list = list(pitch_deg_list)
    if mode == "minus" and project_distance_calculate is not None:
        return project_distance_calculate(pitch_list, camera_height_m=camera_height_m,
                                          camera_pitch_deg=camera_pitch_deg, target_height_m=target_height_m)
    return internal_distance_calculate(pitch_list, camera_height_m=camera_height_m,
                                       camera_pitch_deg=camera_pitch_deg, target_height_m=target_height_m, mode=mode)


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def regression(x, y):
    if len(x) < 2 or len(x) != len(y):
        return None, None, None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    if abs(sxx) < 1e-12:
        return None, None, None
    slope = sxy / sxx
    intercept = my - slope * mx
    corr = sxy / math.sqrt(sxx * syy) if syy > 1e-12 else None
    return slope, intercept, corr


def evaluate_points(points, camera_height_m, camera_pitch_deg, target_height_m, mode) -> EvalResult:
    est_list = calculate_distances([p.raw_pitch_deg for p in points],
                                   camera_height_m, camera_pitch_deg, target_height_m, mode)
    valid_pairs, errors, actuals = [], [], []
    for point, est in zip(points, est_list):
        if est is None:
            continue
        err = est - point.actual_m
        valid_pairs.append((point.actual_m, est))
        actuals.append(point.actual_m)
        errors.append(err)
    slope, intercept, corr = regression(actuals, errors)
    return EvalResult(mode=mode, est_list=est_list, valid_pairs=valid_pairs,
                      mean_error=mean(errors), mean_abs_error=mean([abs(e) for e in errors]),
                      std_error=statistics.stdev(errors) if len(errors) >= 2 else None,
                      slope_err_vs_actual=slope, intercept_err_vs_actual=intercept,
                      corr_err_vs_actual=corr)


def solve_best_delta_height(pitches, actuals, camera_pitch_deg, mode):
    ks, ys = [], []
    for pitch_deg, actual in zip(pitches, actuals):
        total_deg = camera_pitch_deg - pitch_deg if mode == "minus" else camera_pitch_deg + pitch_deg
        t = math.tan(math.radians(total_deg))
        if abs(t) < 1e-9 or (1.0 / t) <= 0:
            return None
        ks.append(1.0 / t)
        ys.append(actual)
    denom = sum(k * k for k in ks)
    return sum(y * k for y, k in zip(ys, ks)) / denom if denom > 1e-12 else None


def evaluate_fit(pitches, actuals, camera_pitch_deg, delta_height_m, mode):
    ests, errors = [], []
    for pitch_deg, actual in zip(pitches, actuals):
        total_deg = camera_pitch_deg - pitch_deg if mode == "minus" else camera_pitch_deg + pitch_deg
        est = delta_height_m / math.tan(math.radians(total_deg))
        ests.append(est)
        errors.append(est - actual)
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    return mae, rmse, sum(errors) / len(errors), ests


def fit_best_params(points, target_height_m, current_pitch_deg, current_height_m, mode):
    if len(points) < 2:
        return None
    pitches = [p.raw_pitch_deg for p in points]
    actuals = [p.actual_m for p in points]
    best, center = None, current_pitch_deg
    for step, span in [(0.10, 10.0), (0.02, 2.0), (0.005, 0.5)]:
        lo, hi = center - span, center + span
        cand = best
        for i in range(int(round((hi - lo) / step)) + 1):
            pitch = lo + i * step
            dh = solve_best_delta_height(pitches, actuals, pitch, mode)
            if dh is None or dh <= 0:
                continue
            ch = target_height_m + dh
            if ch <= target_height_m:
                continue
            mae, rmse, me, ests = evaluate_fit(pitches, actuals, pitch, dh, mode)
            r = FitResult(mode=mode, camera_pitch_deg=pitch, camera_height_m=ch,
                          delta_height_m=dh, mae=mae, rmse=rmse, mean_error=me,
                          est_list=ests, num_used=len(points))
            if cand is None or r.rmse < cand.rmse:
                cand = r
        if cand is None:
            return best
        best, center = cand, cand.camera_pitch_deg
    return best


def fit_pitch_only(points, target_height_m, current_pitch_deg, fixed_camera_height_m, mode):
    if len(points) < 2:
        return None
    pitches = [p.raw_pitch_deg for p in points]
    actuals = [p.actual_m for p in points]
    dh = fixed_camera_height_m - target_height_m
    if dh <= 0:
        return None
    best, center = None, current_pitch_deg
    for step, span in [(0.10, 10.0), (0.02, 2.0), (0.005, 0.5)]:
        lo, hi = center - span, center + span
        cand = best
        for i in range(int(round((hi - lo) / step)) + 1):
            pitch = lo + i * step
            try:
                mae, rmse, me, ests = evaluate_fit(pitches, actuals, pitch, dh, mode)
            except Exception:
                continue
            if any(e is None for e in ests):
                continue
            r = FitResult(mode=mode, camera_pitch_deg=pitch, camera_height_m=fixed_camera_height_m,
                          delta_height_m=dh, mae=mae, rmse=rmse, mean_error=me,
                          est_list=ests, num_used=len(points))
            if cand is None or r.rmse < cand.rmse:
                cand = r
        if cand is None:
            return best
        best, center = cand, cand.camera_pitch_deg
    return best


def fit_height_only(points, target_height_m, fixed_camera_pitch_deg, mode):
    if len(points) < 2:
        return None
    pitches = [p.raw_pitch_deg for p in points]
    actuals = [p.actual_m for p in points]
    dh = solve_best_delta_height(pitches, actuals, fixed_camera_pitch_deg, mode)
    if dh is None or dh <= 0:
        return None
    ch = target_height_m + dh
    mae, rmse, me, ests = evaluate_fit(pitches, actuals, fixed_camera_pitch_deg, dh, mode)
    return FitResult(mode=mode, camera_pitch_deg=fixed_camera_pitch_deg, camera_height_m=ch,
                     delta_height_m=dh, mae=mae, rmse=rmse, mean_error=me,
                     est_list=ests, num_used=len(points))


def summarize_linear_bias(r: EvalResult) -> str:
    if r.slope_err_vs_actual is None or r.corr_err_vs_actual is None:
        return "樣本不足，無法判斷線性偏差。"
    if abs(r.corr_err_vs_actual) < 0.45:
        return "誤差與距離的線性關聯偏弱，較像是固定偏移或量測噪聲。"
    if r.slope_err_vs_actual > 0:
        return "距離越遠，誤差越偏正（遠距高估）。"
    return "距離越遠，誤差越偏負（遠距低估）。"


def analyze_yaw(yaw_center, yaw_robot_left, yaw_robot_right,
                yaw_image_left, yaw_image_right, current_yaw_sign) -> YawAnalysis:
    notes, recs = [], []
    raw_convention = None
    recommended_yaw_sign = None
    center_bias = yaw_center
    offset_delta = None

    if yaw_center is not None:
        tag = "中央偏移小" if abs(yaw_center) <= 1.0 else "中央偏移明顯"
        notes.append(f"{tag}：yaw_center = {yaw_center:+.3f} deg。")

    left  = yaw_robot_left  if yaw_robot_left  is not None else yaw_image_left
    right = yaw_robot_right if yaw_robot_right is not None else yaw_image_right
    basis = "機器人左右" if yaw_robot_left is not None and yaw_robot_right is not None else "畫面左右"

    if left is not None and right is not None:
        if left > right:
            raw_convention = "left_positive_right_negative"
            notes.append(f"[{basis}] raw yaw 趨勢：左側數值較大、右側數值較小。")
            recommended_yaw_sign = -1.0
            recs.append("建議把 CAMERA2_YAW_SIGN 設為 -1.0。")
        elif left < right:
            raw_convention = "left_negative_right_positive"
            notes.append(f"[{basis}] raw yaw 趨勢：左側數值較小、右側數值較大。")
            recommended_yaw_sign = +1.0
            recs.append("建議把 CAMERA2_YAW_SIGN 設為 +1.0。")
        else:
            notes.append(f"[{basis}] 左右量測相同，無法判定 CAMERA2_YAW_SIGN。")

    if recommended_yaw_sign is None:
        recommended_yaw_sign = current_yaw_sign

    if center_bias is not None:
        offset_delta = recommended_yaw_sign * center_bias
        if abs(center_bias) > 1.0:
            recs.append(
                "若不先改 PhotonVision crosshair / 安裝角度，可先用 yaw_center 做軟體補償："
                f" CAMERA2_YAW_OFFSET_DEG += {offset_delta:+.3f} deg"
            )
        else:
            recs.append("中央 yaw 已接近 0，CAMERA2_YAW_OFFSET_DEG 暫時不必因中心偏移而改動。")

    return YawAnalysis(raw_convention=raw_convention, recommended_yaw_sign=recommended_yaw_sign,
                       center_bias_deg=center_bias, offset_delta_deg=offset_delta,
                       notes=notes, recommendations=recs)


def print_point_table(points, est_list):
    print("\n距離點位結果：")
    print("label | actual_m | pitch_deg | yaw_deg | area | estimated_m | error_m")
    print("-" * 78)
    for point, est in zip(points, est_list):
        error = None if est is None else est - point.actual_m
        yaw_txt  = "-" if point.raw_yaw_deg is None else f"{point.raw_yaw_deg:+.3f}"
        area_txt = "-" if point.raw_area    is None else f"{point.raw_area:.3f}"
        est_txt  = "None" if est   is None else f"{est:.3f}"
        err_txt  = "None" if error is None else f"{error:+.3f}"
        print(f"{point.label:>5} | {point.actual_m:8.3f} | {point.raw_pitch_deg:9.3f} | "
              f"{yaw_txt:>7} | {area_txt:>5} | {est_txt:>11} | {err_txt:>7}")


def format_change(name, current, new, unit=""):
    if new is None:
        return f"{name}: 無法估計"
    return f"{name}: {current:.6g} -> {new:.6g} ({new-current:+.6g}{unit})"


def pick_recommended_fit(fit_minus, fit_plus):
    candidates = []
    if fit_minus is not None:
        candidates.append(("minus", fit_minus.rmse, fit_minus))
    if fit_plus is not None:
        candidates.append(("plus", fit_plus.rmse, fit_plus))
    if not candidates:
        return "minus", None
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0], candidates[0][2]


def main():
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    main_py_path = os.path.join(script_dir, "main_final.py")
    if not os.path.exists(main_py_path):
        main_py_path = os.path.join(script_dir, "main_cam2only.py")
    defaults = load_main_defaults(main_py_path)

    print("=" * 78)
    print("CAM2 單鏡頭參數校正工具（數字輸入版）")
    print("用途：用實測距離 + raw pitch/yaw，檢查 main_final.py 內 CAMERA2_* 應修改的常數。")
    print("本工具只需要 Camera2；不需要 Pose2d，也不需要 CAM1。")
    print("=" * 78)
    print()
    print("建議測試流程：")
    print("1. 先做 YAW 方向檢查：中心、機器人左、機器人右（Camera2 視角）。")
    print("2. 再做 4~6 個距離點：盡量讓球在畫面水平中心附近。")
    print("3. 每個點記 actual distance、raw pitch、raw yaw、raw area。")
    print("4. 測完後直接看最後的『main_final.py 修改建議』。")
    print()
    print("⚠ 注意：Camera2 在機器人右側，安裝鏡像。yaw 正負慣例")
    print("  可能與 Camera1 相同，也可能相反。必須以本工具的 YAW 方向測試確認。")
    print()

    print("=== A. 載入 / 確認現用參數 ===")
    camera_height_m      = ask_float("CAMERA_HEIGHT_M",          defaults["CAMERA_HEIGHT_M"])
    camera_pitch_deg     = ask_float("CAMERA_PITCH_DEG",         defaults["CAMERA_PITCH_DEG"])
    target_height_m      = ask_float("TARGET_HEIGHT_M",          defaults["TARGET_HEIGHT_M"])
    camera2_yaw_offset   = ask_float("CAMERA2_YAW_OFFSET_DEG",   defaults["CAMERA2_YAW_OFFSET_DEG"])
    yaw_sign             = ask_float("CAMERA2_YAW_SIGN",         defaults["CAMERA2_YAW_SIGN"])

    print()
    print("=== B. YAW 方向檢查（可跳過，直接 Enter） ===")
    print("固定球的大致距離不要變，只移動球的左右位置。")
    yaw_center      = ask_float("球在 Camera2 畫面正中央時的 raw yaw_deg",    allow_blank=True)
    yaw_robot_left  = ask_float("球移到『機器人左方』時的 raw yaw_deg",       allow_blank=True)
    yaw_robot_right = ask_float("球移到『機器人右方』時的 raw yaw_deg",       allow_blank=True)
    yaw_image_left  = ask_float("球在『Camera2 畫面左方』時的 raw yaw_deg",   allow_blank=True)
    yaw_image_right = ask_float("球在『Camera2 畫面右方』時的 raw yaw_deg",   allow_blank=True)

    yaw_info = analyze_yaw(yaw_center, yaw_robot_left, yaw_robot_right,
                           yaw_image_left, yaw_image_right,
                           yaw_sign if yaw_sign is not None else -1.0)

    print()
    print("=== C. 距離 / PITCH 檢查（至少 4 點，建議 5~6 點） ===")
    print("輸入空白 actual_distance 即結束。")

    points = []
    idx = 1
    while True:
        print(f"\n--- 點位 {idx} ---")
        actual = ask_float("actual_distance_m", allow_blank=True)
        if actual is None:
            break
        raw_pitch = ask_float("raw_pitch_deg")
        raw_yaw   = ask_float("raw_yaw_deg",  allow_blank=True)
        raw_area  = ask_float("raw_area",      allow_blank=True)
        label     = ask_str("label", f"P{idx}")
        points.append(TestPoint(label=label, actual_m=float(actual),
                                raw_pitch_deg=float(raw_pitch),
                                raw_yaw_deg=raw_yaw, raw_area=raw_area))
        idx += 1

    if not points:
        print("\n沒有輸入任何距離點，程式結束。")
        return

    cur_eval  = evaluate_points(points, camera_height_m, camera_pitch_deg, target_height_m, "minus")
    alt_eval  = evaluate_points(points, camera_height_m, camera_pitch_deg, target_height_m, "plus")

    print_point_table(points, cur_eval.est_list)

    fit_minus        = fit_best_params(points, target_height_m, camera_pitch_deg, camera_height_m, "minus")
    fit_plus         = fit_best_params(points, target_height_m, camera_pitch_deg, camera_height_m, "plus")
    pitch_only_minus = fit_pitch_only(points, target_height_m, camera_pitch_deg, camera_height_m, "minus")
    height_only_minus= fit_height_only(points, target_height_m, camera_pitch_deg, "minus")
    pitch_only_plus  = fit_pitch_only(points, target_height_m, camera_pitch_deg, camera_height_m, "plus")
    height_only_plus = fit_height_only(points, target_height_m, camera_pitch_deg, "plus")
    rec_mode, rec_fit = pick_recommended_fit(fit_minus, fit_plus)

    print("\n" + "=" * 78)
    print("結果統整")
    print("=" * 78)

    print("\n[YAW 分析]")
    for note in yaw_info.notes or ["- 未輸入足夠的 YAW 資料，無法判斷 CAMERA2_YAW_SIGN。"]:
        print("-", note)

    print("\n[目前距離模型（減號版本）]")
    if cur_eval.mean_error is not None:
        print(f"- mean_error     = {cur_eval.mean_error:+.4f} m")
        print(f"- mean_abs_error = {cur_eval.mean_abs_error:.4f} m")
        if cur_eval.std_error is not None:
            print(f"- std_error      = {cur_eval.std_error:.4f} m")
        print(f"- 線性偏差判讀    = {summarize_linear_bias(cur_eval)}")
    else:
        print("- 目前參數下沒有有效距離結果。")

    print("\n[改用加號公式的比較]")
    if alt_eval.mean_error is not None:
        print(f"- mean_abs_error = {alt_eval.mean_abs_error:.4f} m")
        print(f"- mean_error     = {alt_eval.mean_error:+.4f} m")
    else:
        print("- 該公式下沒有有效距離結果。")

    print("\n[最佳擬合]")
    if pitch_only_minus:
        print(f"- minus 公式，只調 pitch：pitch={pitch_only_minus.camera_pitch_deg:.3f} deg, rmse={pitch_only_minus.rmse:.4f} m")
    if height_only_minus:
        print(f"- minus 公式，只調 height：height={height_only_minus.camera_height_m:.4f} m, rmse={height_only_minus.rmse:.4f} m")
    if fit_minus:
        print(f"- minus 公式，pitch+height：pitch={fit_minus.camera_pitch_deg:.3f}, height={fit_minus.camera_height_m:.4f}, rmse={fit_minus.rmse:.4f} m")
    if pitch_only_plus:
        print(f"- plus  公式，只調 pitch：pitch={pitch_only_plus.camera_pitch_deg:.3f} deg, rmse={pitch_only_plus.rmse:.4f} m")
    if height_only_plus:
        print(f"- plus  公式，只調 height：height={height_only_plus.camera_height_m:.4f} m, rmse={height_only_plus.rmse:.4f} m")
    if fit_plus:
        print(f"- plus  公式，pitch+height：pitch={fit_plus.camera_pitch_deg:.3f}, height={fit_plus.camera_height_m:.4f}, rmse={fit_plus.rmse:.4f} m")

    print("\n[推薦採用]")
    if rec_fit is None:
        print("- 樣本不足，無法給出穩定推薦。")
    else:
        desc = "維持目前減號公式" if rec_mode == "minus" else "把距離公式改成加號版本"
        print(f"- 建議：{desc}")
        print(f"- 推薦 CAMERA_PITCH_DEG ≈ {rec_fit.camera_pitch_deg:.3f} deg")
        print(f"- 推薦 CAMERA_HEIGHT_M  ≈ {rec_fit.camera_height_m:.4f} m")
        print(f"- 預期 RMSE ≈ {rec_fit.rmse:.4f} m")

    print("\n" + "=" * 78)
    print("main_final.py 修改建議（CAMERA2 專用）")
    print("=" * 78)

    # YAW_SIGN
    if yaw_info.recommended_yaw_sign is not None and abs(yaw_info.recommended_yaw_sign - (yaw_sign or -1.0)) > 1e-9:
        print("-", format_change("CAMERA2_YAW_SIGN", yaw_sign, yaw_info.recommended_yaw_sign))
    else:
        print(f"- CAMERA2_YAW_SIGN: 暫時維持 {yaw_sign:.6g}")

    # YAW_OFFSET
    if yaw_info.offset_delta_deg is not None and abs(yaw_info.offset_delta_deg) > 0.25:
        new_offset = camera2_yaw_offset + yaw_info.offset_delta_deg
        print("-", format_change("CAMERA2_YAW_OFFSET_DEG", camera2_yaw_offset, new_offset, unit=" deg"))
        print(f"  補充：CAMERA2_YAW_OFFSET_DEG += {yaw_info.offset_delta_deg:+.3f} deg（中心偏移補償）")
    else:
        print(f"- CAMERA2_YAW_OFFSET_DEG: 暫時維持 {camera2_yaw_offset:.6g}")

    # PITCH / HEIGHT
    if rec_fit is not None:
        pf = pitch_only_minus if rec_mode == "minus" else pitch_only_plus
        hf = height_only_minus if rec_mode == "minus" else height_only_plus
        if pf:
            print("-", format_change("CAMERA_PITCH_DEG", camera_pitch_deg, pf.camera_pitch_deg, unit=" deg"))
        else:
            print("-", format_change("CAMERA_PITCH_DEG", camera_pitch_deg, rec_fit.camera_pitch_deg, unit=" deg"))
        if hf and abs(hf.camera_height_m - camera_height_m) <= 0.05:
            print("-", format_change("CAMERA_HEIGHT_M", camera_height_m, hf.camera_height_m, unit=" m"))
        else:
            print(f"- CAMERA_HEIGHT_M: 建議先維持 {camera_height_m:.6g} m（需超過 5 cm 才建議大幅改動）。")
        if rec_mode == "plus":
            print("- distance_utils.py: 建議把 total_deg = camera_pitch_deg - pitch_deg 改成 camera_pitch_deg + pitch_deg。")
        else:
            print("- distance_utils.py: 維持 total_deg = camera_pitch_deg - pitch_deg。")
    else:
        print("- 樣本不足，無法對 CAMERA_PITCH_DEG / CAMERA_HEIGHT_M 給出穩定估計。")

    print("\n[額外說明]")
    for rec in yaw_info.recommendations:
        print("-", rec)
    if cur_eval.mean_error is not None:
        if cur_eval.mean_error > 0.08:
            print("- 現況偏高估：CAMERA_PITCH_DEG 可能偏小，或 CAMERA_HEIGHT_M 可能偏高。")
        elif cur_eval.mean_error < -0.08:
            print("- 現況偏低估：CAMERA_PITCH_DEG 可能偏大，或 CAMERA_HEIGHT_M 可能偏低。")
        else:
            print("- 現況平均偏差不大，主要看遠距離是否仍有線性偏差。")

    print("\n完成。把上方修改建議套回 main_final.py 的 CAMERA2_* 常數後，重新執行 main_final.py 驗證。")


if __name__ == "__main__":
    main()