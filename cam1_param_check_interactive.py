from __future__ import annotations

import ast
import math
import os
import re
import statistics
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Try to use the project's current distance model first.
# Fallback to an internal implementation if import is unavailable.
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
    mode: str  # "minus" or "plus"
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


MAIN_DEFAULTS = {
    "CAMERA_HEIGHT_M": 0.55,
    "CAMERA_PITCH_DEG": 25.0,
    "TARGET_HEIGHT_M": 0.075,
    "CAMERA1_YAW_OFFSET_DEG": 35.0,
    "YAW_SIGN": 1.0,
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
        shown_default = None if default is None else f"{default:.6g}"
        if shown_default is None:
            s = input(f"{prompt}: ").strip()
        else:
            s = input(f"{prompt} [{shown_default}]: ").strip()
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
        if mode == "minus":
            total_deg = camera_pitch_deg - pitch_deg
        elif mode == "plus":
            total_deg = camera_pitch_deg + pitch_deg
        else:
            raise ValueError(f"Unknown mode: {mode}")

        total_rad = math.radians(total_deg)
        tanvalue = math.tan(total_rad)
        if abs(tanvalue) < 1e-9:
            out.append(None)
            continue
        distance = delta_height / tanvalue
        if distance <= 0:
            out.append(None)
        else:
            out.append(distance)
    return out


def calculate_distances(
    pitch_deg_list: Iterable[float],
    camera_height_m: float,
    camera_pitch_deg: float,
    target_height_m: float,
    mode: str = "minus",
) -> List[Optional[float]]:
    pitch_list = list(pitch_deg_list)
    if mode == "minus" and project_distance_calculate is not None:
        return project_distance_calculate(
            pitch_list,
            camera_height_m=camera_height_m,
            camera_pitch_deg=camera_pitch_deg,
            target_height_m=target_height_m,
        )
    return internal_distance_calculate(
        pitch_list,
        camera_height_m=camera_height_m,
        camera_pitch_deg=camera_pitch_deg,
        target_height_m=target_height_m,
        mode=mode,
    )


def mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def regression(x: List[float], y: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
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
    corr = None
    if syy > 1e-12:
        corr = sxy / math.sqrt(sxx * syy)
    return slope, intercept, corr


def evaluate_points(
    points: List[TestPoint],
    camera_height_m: float,
    camera_pitch_deg: float,
    target_height_m: float,
    mode: str,
) -> EvalResult:
    pitches = [p.raw_pitch_deg for p in points]
    est_list = calculate_distances(
        pitches,
        camera_height_m=camera_height_m,
        camera_pitch_deg=camera_pitch_deg,
        target_height_m=target_height_m,
        mode=mode,
    )

    valid_pairs: List[Tuple[float, float]] = []
    errors: List[float] = []
    actuals: List[float] = []
    for point, est in zip(points, est_list):
        if est is None:
            continue
        actual = point.actual_m
        err = est - actual
        valid_pairs.append((actual, est))
        actuals.append(actual)
        errors.append(err)

    mean_error = mean(errors)
    mean_abs_error = mean([abs(e) for e in errors])
    std_error = statistics.stdev(errors) if len(errors) >= 2 else None
    slope, intercept, corr = regression(actuals, errors)

    return EvalResult(
        mode=mode,
        est_list=est_list,
        valid_pairs=valid_pairs,
        mean_error=mean_error,
        mean_abs_error=mean_abs_error,
        std_error=std_error,
        slope_err_vs_actual=slope,
        intercept_err_vs_actual=intercept,
        corr_err_vs_actual=corr,
    )


def solve_best_delta_height(pitches: List[float], actuals: List[float], camera_pitch_deg: float, mode: str) -> Optional[float]:
    ks: List[float] = []
    ys: List[float] = []
    for pitch_deg, actual in zip(pitches, actuals):
        total_deg = camera_pitch_deg - pitch_deg if mode == "minus" else camera_pitch_deg + pitch_deg
        total_rad = math.radians(total_deg)
        tanvalue = math.tan(total_rad)
        if abs(tanvalue) < 1e-9:
            return None
        k = 1.0 / tanvalue
        if k <= 0:
            return None
        ks.append(k)
        ys.append(actual)
    denom = sum(k * k for k in ks)
    if denom <= 1e-12:
        return None
    return sum(y * k for y, k in zip(ys, ks)) / denom


def evaluate_fit(pitches: List[float], actuals: List[float], camera_pitch_deg: float, delta_height_m: float, mode: str) -> Tuple[float, float, float, List[float]]:
    ests: List[float] = []
    errors: List[float] = []
    for pitch_deg, actual in zip(pitches, actuals):
        total_deg = camera_pitch_deg - pitch_deg if mode == "minus" else camera_pitch_deg + pitch_deg
        total_rad = math.radians(total_deg)
        tanvalue = math.tan(total_rad)
        est = delta_height_m / tanvalue
        ests.append(est)
        errors.append(est - actual)
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    mean_error = sum(errors) / len(errors)
    return mae, rmse, mean_error, ests


def fit_best_params(
    points: List[TestPoint],
    target_height_m: float,
    current_pitch_deg: float,
    current_height_m: float,
    mode: str,
) -> Optional[FitResult]:
    if len(points) < 2:
        return None

    pitches = [p.raw_pitch_deg for p in points]
    actuals = [p.actual_m for p in points]

    best: Optional[FitResult] = None
    center = current_pitch_deg

    for step, span in [(0.10, 10.0), (0.02, 2.0), (0.005, 0.5)]:
        lo = center - span
        hi = center + span
        n = int(round((hi - lo) / step))
        candidate_best: Optional[FitResult] = best
        for i in range(n + 1):
            pitch = lo + i * step
            delta_h = solve_best_delta_height(pitches, actuals, pitch, mode)
            if delta_h is None or delta_h <= 0:
                continue
            camera_height = target_height_m + delta_h
            if camera_height <= target_height_m:
                continue
            mae, rmse, mean_error, ests = evaluate_fit(pitches, actuals, pitch, delta_h, mode)
            candidate = FitResult(
                mode=mode,
                camera_pitch_deg=pitch,
                camera_height_m=camera_height,
                delta_height_m=delta_h,
                mae=mae,
                rmse=rmse,
                mean_error=mean_error,
                est_list=ests,
                num_used=len(points),
            )
            if candidate_best is None or candidate.rmse < candidate_best.rmse:
                candidate_best = candidate
        if candidate_best is None:
            return best
        best = candidate_best
        center = best.camera_pitch_deg

    return best




def fit_pitch_only(
    points: List[TestPoint],
    target_height_m: float,
    current_pitch_deg: float,
    fixed_camera_height_m: float,
    mode: str,
) -> Optional[FitResult]:
    if len(points) < 2:
        return None
    pitches = [p.raw_pitch_deg for p in points]
    actuals = [p.actual_m for p in points]
    delta_height_m = fixed_camera_height_m - target_height_m
    if delta_height_m <= 0:
        return None

    best: Optional[FitResult] = None
    center = current_pitch_deg
    for step, span in [(0.10, 10.0), (0.02, 2.0), (0.005, 0.5)]:
        lo = center - span
        hi = center + span
        n = int(round((hi - lo) / step))
        candidate_best = best
        for i in range(n + 1):
            pitch = lo + i * step
            try:
                mae, rmse, mean_error, ests = evaluate_fit(pitches, actuals, pitch, delta_height_m, mode)
            except Exception:
                continue
            if any((e is None) for e in ests):
                continue
            candidate = FitResult(
                mode=mode,
                camera_pitch_deg=pitch,
                camera_height_m=fixed_camera_height_m,
                delta_height_m=delta_height_m,
                mae=mae,
                rmse=rmse,
                mean_error=mean_error,
                est_list=ests,
                num_used=len(points),
            )
            if candidate_best is None or candidate.rmse < candidate_best.rmse:
                candidate_best = candidate
        if candidate_best is None:
            return best
        best = candidate_best
        center = best.camera_pitch_deg
    return best


def fit_height_only(
    points: List[TestPoint],
    target_height_m: float,
    fixed_camera_pitch_deg: float,
    mode: str,
) -> Optional[FitResult]:
    if len(points) < 2:
        return None
    pitches = [p.raw_pitch_deg for p in points]
    actuals = [p.actual_m for p in points]
    delta_h = solve_best_delta_height(pitches, actuals, fixed_camera_pitch_deg, mode)
    if delta_h is None or delta_h <= 0:
        return None
    camera_height_m = target_height_m + delta_h
    mae, rmse, mean_error, ests = evaluate_fit(pitches, actuals, fixed_camera_pitch_deg, delta_h, mode)
    return FitResult(
        mode=mode,
        camera_pitch_deg=fixed_camera_pitch_deg,
        camera_height_m=camera_height_m,
        delta_height_m=delta_h,
        mae=mae,
        rmse=rmse,
        mean_error=mean_error,
        est_list=ests,
        num_used=len(points),
    )


def summarize_linear_bias(eval_result: EvalResult) -> str:
    slope = eval_result.slope_err_vs_actual
    corr = eval_result.corr_err_vs_actual
    if slope is None or corr is None:
        return "樣本不足，無法判斷是否存在線性偏差。"

    if abs(corr) < 0.45:
        return "誤差與距離的線性關聯偏弱，較像是固定偏移或量測噪聲。"
    if slope > 0:
        return "存在正斜率：距離越遠，誤差越偏正，代表遠距離越容易高估。"
    return "存在負斜率：距離越遠，誤差越偏負，代表遠距離越容易低估。"


def analyze_yaw(
    yaw_center: Optional[float],
    yaw_robot_left: Optional[float],
    yaw_robot_right: Optional[float],
    yaw_image_left: Optional[float],
    yaw_image_right: Optional[float],
    current_yaw_sign: float,
) -> YawAnalysis:
    notes: List[str] = []
    recs: List[str] = []
    raw_convention: Optional[str] = None
    recommended_yaw_sign: Optional[float] = None
    center_bias = yaw_center
    offset_delta = None

    if yaw_center is not None:
        if abs(yaw_center) <= 1.0:
            notes.append(f"中央偏移小：yaw_center = {yaw_center:+.3f} deg。")
        else:
            notes.append(f"中央偏移明顯：yaw_center = {yaw_center:+.3f} deg。")

    # Prefer robot-left / robot-right because that is the meaningful direction for field geometry.
    left = yaw_robot_left if yaw_robot_left is not None else yaw_image_left
    right = yaw_robot_right if yaw_robot_right is not None else yaw_image_right
    basis = "機器人左右" if yaw_robot_left is not None and yaw_robot_right is not None else "畫面左右"

    if left is not None and right is not None:
        if left > right:
            raw_convention = "left_positive_right_negative"
            notes.append(f"[{basis}] raw yaw 趨勢：左側數值較大、右側數值較小。")
            recommended_yaw_sign = -1.0
            recs.append("若沿用 bearing_robot_deg = camera_yaw_offset_deg - YAW_SIGN * yaw_deg，建議把 YAW_SIGN 設為 -1.0。")
        elif left < right:
            raw_convention = "left_negative_right_positive"
            notes.append(f"[{basis}] raw yaw 趨勢：左側數值較小、右側數值較大。")
            recommended_yaw_sign = +1.0
            recs.append("若沿用 bearing_robot_deg = camera_yaw_offset_deg - YAW_SIGN * yaw_deg，建議把 YAW_SIGN 設為 +1.0。")
        else:
            notes.append(f"[{basis}] 左右量測相同，無法判定 YAW_SIGN。")

    if recommended_yaw_sign is None:
        recommended_yaw_sign = current_yaw_sign

    if center_bias is not None:
        offset_delta = recommended_yaw_sign * center_bias
        if abs(center_bias) > 1.0:
            recs.append(
                "若不先改 PhotonVision crosshair / 安裝角度，可先用 yaw_center 做軟體補償："
                f" CAMERA1_YAW_OFFSET_DEG += {offset_delta:+.3f} deg"
            )
        else:
            recs.append("中央 yaw 已接近 0，CAMERA1_YAW_OFFSET_DEG 暫時不必因中心偏移而改動。")

    return YawAnalysis(
        raw_convention=raw_convention,
        recommended_yaw_sign=recommended_yaw_sign,
        center_bias_deg=center_bias,
        offset_delta_deg=offset_delta,
        notes=notes,
        recommendations=recs,
    )


def print_point_table(points: List[TestPoint], est_list: List[Optional[float]]) -> None:
    print("\n距離點位結果：")
    print("label | actual_m | pitch_deg | yaw_deg | area | estimated_m | error_m")
    print("-" * 78)
    for point, est in zip(points, est_list):
        error = None if est is None else est - point.actual_m
        yaw_txt = "-" if point.raw_yaw_deg is None else f"{point.raw_yaw_deg:+.3f}"
        area_txt = "-" if point.raw_area is None else f"{point.raw_area:.3f}"
        est_txt = "None" if est is None else f"{est:.3f}"
        err_txt = "None" if error is None else f"{error:+.3f}"
        print(
            f"{point.label:>5} | {point.actual_m:8.3f} | {point.raw_pitch_deg:9.3f} | {yaw_txt:>7} | {area_txt:>5} | {est_txt:>11} | {err_txt:>7}"
        )


def format_change(name: str, current: float, new: Optional[float], unit: str = "") -> str:
    if new is None:
        return f"{name}: 無法估計"
    delta = new - current
    return f"{name}: {current:.6g} -> {new:.6g} ({delta:+.6g}{unit})"


def pick_recommended_fit(current_eval: EvalResult, fit_minus: Optional[FitResult], fit_plus: Optional[FitResult]) -> Tuple[str, Optional[FitResult]]:
    candidates: List[Tuple[str, float, Optional[FitResult]]] = []
    if fit_minus is not None:
        candidates.append(("minus", fit_minus.rmse, fit_minus))
    if fit_plus is not None:
        candidates.append(("plus", fit_plus.rmse, fit_plus))
    if not candidates:
        return "minus", None
    candidates.sort(key=lambda item: item[1])
    return candidates[0][0], candidates[0][2]


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_py_path = os.path.join(script_dir, "main.py")
    defaults = load_main_defaults(main_py_path)

    print("=" * 78)
    print("CAM1 單鏡頭參數校正工具（數字輸入版）")
    print("用途：用實測距離 + raw pitch/yaw，檢查 main.py 內應該修改哪些常數。")
    print("本工具只需要 Camera1；不需要 Pose2d，也不需要 CAM2。")
    print("=" * 78)
    print()
    print("建議測試流程：")
    print("1. 先做 YAW 方向檢查：中心、機器人左、機器人右。")
    print("2. 再做 4~6 個距離點：盡量讓球在畫面水平中心附近。")
    print("3. 每個點記 actual distance、raw pitch、raw yaw、raw area。")
    print("4. 測完後直接看本工具最後的『main.py 修改建議』。")
    print()

    print("=== A. 載入 / 確認 main.py 現用參數 ===")
    camera_height_m = ask_float("CAMERA_HEIGHT_M", defaults["CAMERA_HEIGHT_M"])
    camera_pitch_deg = ask_float("CAMERA_PITCH_DEG", defaults["CAMERA_PITCH_DEG"])
    target_height_m = ask_float("TARGET_HEIGHT_M", defaults["TARGET_HEIGHT_M"])
    camera1_yaw_offset_deg = ask_float("CAMERA1_YAW_OFFSET_DEG", defaults["CAMERA1_YAW_OFFSET_DEG"])
    yaw_sign = ask_float("YAW_SIGN", defaults["YAW_SIGN"])

    print()
    print("=== B. YAW 方向檢查（可跳過，直接 Enter） ===")
    print("先固定球的大致距離不要變。")
    yaw_center = ask_float("球在畫面正中央 / 你認為中心線附近時的 raw yaw_deg", allow_blank=True)
    yaw_robot_left = ask_float("球移到『機器人左方』時的 raw yaw_deg", allow_blank=True)
    yaw_robot_right = ask_float("球移到『機器人右方』時的 raw yaw_deg", allow_blank=True)
    yaw_image_left = ask_float("球在『畫面左方』時的 raw yaw_deg", allow_blank=True)
    yaw_image_right = ask_float("球在『畫面右方』時的 raw yaw_deg", allow_blank=True)

    yaw_info = analyze_yaw(
        yaw_center=yaw_center,
        yaw_robot_left=yaw_robot_left,
        yaw_robot_right=yaw_robot_right,
        yaw_image_left=yaw_image_left,
        yaw_image_right=yaw_image_right,
        current_yaw_sign=yaw_sign if yaw_sign is not None else 1.0,
    )

    print()
    print("=== C. 距離 / PITCH 檢查（至少 4 點，建議 5~6 點） ===")
    print("輸入空白 actual_distance 即結束。")

    points: List[TestPoint] = []
    idx = 1
    while True:
        print(f"\n--- 點位 {idx} ---")
        actual = ask_float("actual_distance_m", allow_blank=True)
        if actual is None:
            break
        raw_pitch = ask_float("raw_pitch_deg")
        raw_yaw = ask_float("raw_yaw_deg", allow_blank=True)
        raw_area = ask_float("raw_area", allow_blank=True)
        label_default = f"P{idx}"
        label = ask_str("label", label_default)
        points.append(
            TestPoint(
                label=label,
                actual_m=float(actual),
                raw_pitch_deg=float(raw_pitch),
                raw_yaw_deg=raw_yaw,
                raw_area=raw_area,
            )
        )
        idx += 1

    if not points:
        print("\n沒有輸入任何距離點，程式結束。")
        return

    current_eval = evaluate_points(
        points,
        camera_height_m=camera_height_m,
        camera_pitch_deg=camera_pitch_deg,
        target_height_m=target_height_m,
        mode="minus",
    )
    alt_eval = evaluate_points(
        points,
        camera_height_m=camera_height_m,
        camera_pitch_deg=camera_pitch_deg,
        target_height_m=target_height_m,
        mode="plus",
    )

    print_point_table(points, current_eval.est_list)

    fit_minus = fit_best_params(points, target_height_m, camera_pitch_deg, camera_height_m, mode="minus")
    fit_plus = fit_best_params(points, target_height_m, camera_pitch_deg, camera_height_m, mode="plus")
    pitch_only_minus = fit_pitch_only(points, target_height_m, camera_pitch_deg, camera_height_m, mode="minus")
    height_only_minus = fit_height_only(points, target_height_m, camera_pitch_deg, mode="minus")
    pitch_only_plus = fit_pitch_only(points, target_height_m, camera_pitch_deg, camera_height_m, mode="plus")
    height_only_plus = fit_height_only(points, target_height_m, camera_pitch_deg, mode="plus")
    recommended_mode, recommended_fit = pick_recommended_fit(current_eval, fit_minus, fit_plus)

    print("\n" + "=" * 78)
    print("結果統整")
    print("=" * 78)

    print("\n[YAW 分析]")
    if yaw_info.notes:
        for note in yaw_info.notes:
            print("-", note)
    else:
        print("- 未輸入足夠的 YAW 資料，無法判斷 YAW_SIGN。")

    print("\n[目前 main.py 距離模型（distance_utils 的減號版本）]")
    if current_eval.mean_error is not None:
        print(f"- mean_error     = {current_eval.mean_error:+.4f} m   (estimated - actual)")
        print(f"- mean_abs_error = {current_eval.mean_abs_error:.4f} m")
        if current_eval.std_error is not None:
            print(f"- std_error      = {current_eval.std_error:.4f} m")
        if current_eval.slope_err_vs_actual is not None:
            print(f"- err slope      = {current_eval.slope_err_vs_actual:+.4f} m/m")
        if current_eval.corr_err_vs_actual is not None:
            print(f"- err corr       = {current_eval.corr_err_vs_actual:+.4f}")
        print(f"- 線性偏差判讀    = {summarize_linear_bias(current_eval)}")
    else:
        print("- 目前參數下沒有有效距離結果。")

    print("\n[同一組常數，但把公式改成 total_deg = camera_pitch_deg + pitch_deg 後的比較]")
    if alt_eval.mean_error is not None:
        print(f"- mean_abs_error = {alt_eval.mean_abs_error:.4f} m")
        print(f"- mean_error     = {alt_eval.mean_error:+.4f} m")
    else:
        print("- 該公式下沒有有效距離結果。")

    print("\n[最佳擬合（只改常數，不改資料）]")
    if pitch_only_minus is not None:
        print(f"- minus 公式，固定 height 只調 pitch：pitch={pitch_only_minus.camera_pitch_deg:.3f} deg, rmse={pitch_only_minus.rmse:.4f} m")
    if height_only_minus is not None:
        print(f"- minus 公式，固定 pitch 只調 height：height={height_only_minus.camera_height_m:.4f} m, rmse={height_only_minus.rmse:.4f} m")
    if fit_minus is not None:
        print(f"- minus 公式，pitch+height 同時擬合：pitch={fit_minus.camera_pitch_deg:.3f} deg, height={fit_minus.camera_height_m:.4f} m, rmse={fit_minus.rmse:.4f} m")
    else:
        print("- minus 公式：無法擬合")

    if pitch_only_plus is not None:
        print(f"- plus  公式，固定 height 只調 pitch：pitch={pitch_only_plus.camera_pitch_deg:.3f} deg, rmse={pitch_only_plus.rmse:.4f} m")
    if height_only_plus is not None:
        print(f"- plus  公式，固定 pitch 只調 height：height={height_only_plus.camera_height_m:.4f} m, rmse={height_only_plus.rmse:.4f} m")
    if fit_plus is not None:
        print(f"- plus  公式，pitch+height 同時擬合：pitch={fit_plus.camera_pitch_deg:.3f} deg, height={fit_plus.camera_height_m:.4f} m, rmse={fit_plus.rmse:.4f} m")
    else:
        print("- plus  公式：無法擬合")

    print("\n[推薦採用]")
    if recommended_fit is None:
        print("- 樣本不足，無法給出穩定推薦。")
    else:
        mode_desc = "維持目前減號公式" if recommended_mode == "minus" else "把距離公式改成加號版本"
        print(f"- 建議：{mode_desc}")
        print(f"- 推薦參數：CAMERA_PITCH_DEG ≈ {recommended_fit.camera_pitch_deg:.3f} deg")
        print(f"- 推薦參數：CAMERA_HEIGHT_M  ≈ {recommended_fit.camera_height_m:.4f} m")
        print(f"- 推薦預期：RMSE ≈ {recommended_fit.rmse:.4f} m")

    print("\n" + "=" * 78)
    print("main.py 修改建議")
    print("=" * 78)

    if yaw_info.recommended_yaw_sign is not None and abs(yaw_info.recommended_yaw_sign - yaw_sign) > 1e-9:
        print("-", format_change("YAW_SIGN", yaw_sign, yaw_info.recommended_yaw_sign))
    else:
        print(f"- YAW_SIGN: 暫時維持 {yaw_sign:.6g}")

    if yaw_info.offset_delta_deg is not None and abs(yaw_info.offset_delta_deg) > 0.25:
        new_offset = camera1_yaw_offset_deg + yaw_info.offset_delta_deg
        print("-", format_change("CAMERA1_YAW_OFFSET_DEG", camera1_yaw_offset_deg, new_offset, unit=" deg"))
        print(f"  補充：這是用 yaw_center 做中心偏移補償，等於 CAMERA1_YAW_OFFSET_DEG += {yaw_info.offset_delta_deg:+.3f} deg")
    else:
        print(f"- CAMERA1_YAW_OFFSET_DEG: 暫時維持 {camera1_yaw_offset_deg:.6g}")

    if recommended_fit is not None:
        preferred_pitch_fit = pitch_only_minus if recommended_mode == "minus" else pitch_only_plus
        preferred_height_fit = height_only_minus if recommended_mode == "minus" else height_only_plus

        if preferred_pitch_fit is not None:
            print("-", format_change("CAMERA_PITCH_DEG", camera_pitch_deg, preferred_pitch_fit.camera_pitch_deg, unit=" deg"))
        else:
            print("-", format_change("CAMERA_PITCH_DEG", camera_pitch_deg, recommended_fit.camera_pitch_deg, unit=" deg"))

        if preferred_height_fit is not None and abs(preferred_height_fit.camera_height_m - camera_height_m) <= 0.05:
            print("-", format_change("CAMERA_HEIGHT_M", camera_height_m, preferred_height_fit.camera_height_m, unit=" m"))
        else:
            print(f"- CAMERA_HEIGHT_M: 建議先維持 {camera_height_m:.6g} m，並重新實測安裝高度；目前資料不建議直接大改高度常數。")

        print(f"- TARGET_HEIGHT_M: 建議先維持 {target_height_m:.6g} m；除非你確認 PhotonVision 取到的不是球心高度。")

        if abs(recommended_fit.camera_height_m - camera_height_m) > 0.05 or (preferred_height_fit is not None and abs(preferred_height_fit.camera_height_m - camera_height_m) > 0.05):
            print("- 注意：資料顯示若只靠改高度才能修正，所需變動已超過 5 cm。若你的實際安裝高度量得很準，這通常代表：")
            print("  1) CAMERA_PITCH_DEG 還沒量準，或")
            print("  2) PhotonVision 的框中心不是你假設的球心，或")
            print("  3) 測試點沒有放在畫面水平中心附近。")
            print("  所以實務上建議：先改 CAMERA_PITCH_DEG，再確認球是否在畫面中央，最後才微調 CAMERA_HEIGHT_M。")

        if recommended_mode == "plus":
            print("- distance_utils.py: 建議把 total_deg = camera_pitch_deg - pitch_deg 改成 total_deg = camera_pitch_deg + pitch_deg 後再重測。")
        else:
            print("- distance_utils.py: 暫時維持 total_deg = camera_pitch_deg - pitch_deg。")
    else:
        print("- 樣本不足，無法對 CAMERA_PITCH_DEG / CAMERA_HEIGHT_M 給出穩定估計。")

    print("\n[額外說明]")
    for rec in yaw_info.recommendations:
        print("-", rec)

    if current_eval.mean_error is not None:
        if current_eval.mean_error > 0.08:
            print("- 現況偏高估：通常代表 CAMERA_PITCH_DEG 可能偏小，或 CAMERA_HEIGHT_M 可能偏高。")
        elif current_eval.mean_error < -0.08:
            print("- 現況偏低估：通常代表 CAMERA_PITCH_DEG 可能偏大，或 CAMERA_HEIGHT_M 可能偏低。")
        else:
            print("- 現況平均偏差不大，主要看遠距離是否仍有線性偏差。")

    print("\n完成。你明天測完後，把最後『main.py 修改建議』直接套回主程式即可。")


if __name__ == "__main__":
    main()
