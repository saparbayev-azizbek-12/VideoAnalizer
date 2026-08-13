from __future__ import annotations
import os
import cv2
import sys
import glob
import argparse
import numpy as np


def _open_source(source: str) -> cv2.VideoCapture:
    try:
        src: int | str = int(source)
    except ValueError:
        src = source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera manbasi ochilmadi: {source}")
    return cap


def _parse_board(board: str) -> tuple[int, int]:
    cols_s, rows_s = board.lower().split("x")
    return int(cols_s), int(rows_s)


def cmd_capture(args: argparse.Namespace) -> None:
    cols, rows = _parse_board(args.board)
    os.makedirs(args.out, exist_ok=True)

    existing = glob.glob(os.path.join(args.out, "calib_*.png"))
    idx = len(existing)

    cap = _open_source(args.source)

    criteria_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preview = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags=criteria_flags)
        if found:
            cv2.drawChessboardCorners(preview, (cols, rows), corners, found)
            cv2.putText(preview, "TAXTA TOPILDI - 's' bilan saqlang", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(preview, "Taxta topilmadi", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(preview, f"Saqlangan: {idx}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Capture (s=save, q=quit)", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            path = os.path.join(args.out, f"calib_{idx:03d}.png")
            cv2.imwrite(path, frame)
            idx += 1
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def cmd_calibrate(args: argparse.Namespace) -> None:
    cols, rows = _parse_board(args.board)
    square = args.square

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []

    images = sorted(glob.glob(os.path.join(args.images, "*.png"))
                     + glob.glob(os.path.join(args.images, "*.jpg")))
    if not images:
        sys.exit(1)

    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    image_size: tuple[int, int] | None = None
    used = 0

    for path in images:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, (cols, rows))
        if not found:
            continue

        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        used += 1

    if used < 5:
        sys.exit(1)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    total_error = 0.0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        total_error += err
    mean_error = total_error / len(objpoints)

    np.savez(args.save,
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             image_size=np.array(image_size),
             mean_error=mean_error)

def cmd_test(args: argparse.Namespace) -> None:
    data = np.load(args.calib)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    w, h = int(data["image_size"][0]), int(data["image_size"][1])

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=args.alpha
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_16SC2
    )

    cap = _open_source(args.source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

        h_disp = 480
        scale = h_disp / frame.shape[0]
        left = cv2.resize(frame, (int(frame.shape[1] * scale), h_disp))
        right = cv2.resize(undistorted, (int(undistorted.shape[1] * scale), h_disp))
        combined = np.hstack([left, right])

        cv2.imshow("Asl (chap)  vs  To'g'irlangan (o'ng) - q=chiqish", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Camera calibration", formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cap = sub.add_parser("capture", help="Checkerboard suratlarni yig'ish")
    p_cap.add_argument("--source", required=True, help="RTSP URL yoki USB device index (masalan 0)")
    p_cap.add_argument("--out", default="calib_images", help="Suratlar saqlanadigan papka")
    p_cap.add_argument("--board", default="9x6", help="Ichki burchak nuqtalari, masalan 9x6")
    p_cap.set_defaults(func=cmd_capture)

    p_cal = sub.add_parser("calibrate", help="Yig'ilgan suratlardan kalibratsiya hisoblash")
    p_cal.add_argument("--images", required=True, help="Suratlar papkasi")
    p_cal.add_argument("--board", default="9x6", help="Ichki burchak nuqtalari, masalan 9x6")
    p_cal.add_argument("--square", type=float, required=True, help="Katak o'lchami (mm)")
    p_cal.add_argument("--save", default="calib.npz", help="Natija saqlanadigan fayl")
    p_cal.set_defaults(func=cmd_calibrate)

    p_test = sub.add_parser("test", help="Natijani jonli oqimda tekshirish")
    p_test.add_argument("--source", required=True, help="RTSP URL yoki USB device index")
    p_test.add_argument("--calib", default="calib.npz", help="calibrate rejimidan chiqqan fayl")
    p_test.add_argument("--alpha", type=float, default=1.0,
                         help="0=qora chetlarni kesib tashla, 1=barcha pikselni saqla")
    p_test.set_defaults(func=cmd_test)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()