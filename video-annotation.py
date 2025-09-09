#------------------------------------------------------------------------------
# Player example. Plays PV and Microphone data that was previously recorded 
# using simple recorder.
#------------------------------------------------------------------------------
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'hl2ss', 'viewer')))

# these hl2ss imports come from the above sys.path.append()
import hl2ss
import hl2ss_io
import hl2ss_3dcv

import numpy as np
import cv2
import cv2.aruco as aruco
import pickle
from utils import *
import time
from drawing_utils import *
from scipy.spatial.transform import Rotation as scipy_R
from Rendering.PyrenderRenderer import EfficientMeshRenderer
import argparse


def mask_to_bbox_xywh(mask):
    """
    From a single-channel boolean or 0/1 mask, compute the axis-aligned bounding
    box in pixel coordinates as [x_min, y_min, width, height].

    Parameters
    ----------
    mask : np.ndarray
        2-D array of shape (H, W). Non-zero values mark the foreground.

    Returns
    -------
    np.ndarray | None
        Array of shape (4,) with dtype np.int32:
        [x_min, y_min, width, height]
        or None if the mask is empty.
    """
    ys, xs = np.where(mask)
    if xs.size == 0:        # empty mask
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Width/height include both end pixels (+1).  
    # Omit the +1 if you need the half-open [x_min, x_max) convention.
    width  = x_max - x_min + 1
    height = y_max - y_min + 1

    return np.array([x_min, y_min, width, height], dtype=np.int32)

def draw_projected_points(img, image_points, color=(0, 255, 0)):
    """
    img: BGR image (modified in place)
    image_points: output of cv2.projectPoints, shape (N,1,2) or (N,2)
    """
    pts = image_points.reshape(-1, 2)  # (N,2)
    for x, y in pts:
        cv2.circle(
            img,
            (int(round(x)), int(round(y))),
            radius=3,
            color=color,
            thickness=-1,              # filled
            lineType=cv2.LINE_AA
        )

def rainbow_colors(n):
    """Return n BGR colors sweeping the hue spectrum."""
    # generate HSV with H from 0..179 (OpenCV hue range), S=255, V=255
    hsv = np.zeros((n, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = np.linspace(0, 179, n, endpoint=True, dtype=np.uint8)  # hue
    hsv[:, 0, 1] = 255  # saturation
    hsv[:, 0, 2] = 255  # value
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # shape (n,1,3)
    return bgr.reshape(n, 3)  # (n,3)

def draw_projected_points_rainbow(img, image_points):
    pts = image_points.reshape(-1, 2)
    colors = rainbow_colors(len(pts))
    for (x, y), (b, g, r) in zip(pts, colors):
        cv2.circle(
            img,
            (int(round(x)), int(round(y))),
            radius=2,
            color=(int(b), int(g), int(r)),
            thickness=-1,
            lineType=cv2.LINE_AA
        )

def make4x4(R, t):
    T = np.eye(4)
    T[0:3,0:3] = R
    T[0:3,3] = t.ravel()

    return T

def getRt(T):
    R = T[0:3,0:3]
    t = (T[0:3,3]).reshape(-1,1)

    return R, t

def apply_mask_overlay(image, mask, color=(255, 0, 0), alpha=0.5):
    """
    Apply a semi-transparent color overlay to an image using a binary mask.

    Parameters
    ----------
    image : np.ndarray
        RGB image, shape (H, W, 3), dtype uint8
    mask : np.ndarray
        Boolean mask, shape (H, W)
    color : tuple
        RGB color for overlay (default: red)
    alpha : float
        Transparency of the overlay (0 = transparent, 1 = solid)
    
    Returns
    -------
    result : np.ndarray
        Image with overlay applied
    """
    overlay = image.copy()
    color_array = np.array(color, dtype=np.float32)

    # Blend only where mask is True
    mask_idx = np.where(mask)
    for c in range(3):  # For R, G, B
        overlay[mask_idx[0], mask_idx[1], c] = (
            (1 - alpha) * image[mask_idx[0], mask_idx[1], c] +
            alpha * color_array[c]
        )

    return overlay.astype(np.uint8)


# Settings --------------------------------------------------------------------

#------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a target directory.")
    parser.add_argument("videopath", type=str, help="Path to the target directory")
    args = parser.parse_args()

    path = os.path.abspath(os.path.expanduser(args.videopath))

    tweak_fname = 'tweak.txt'

    output_dir = os.path.join(path, 'output')

    left_dir = os.path.join(output_dir, 'left')
    mask_dir = os.path.join(output_dir, 'mask')
    pose_dir = os.path.join(output_dir, 'pose')
    rgb_dir = os.path.join(output_dir, 'rgb')
    right_dir = os.path.join(output_dir, 'right')
    vis_dir = os.path.join(output_dir, 'vis')
    bbox_dir = os.path.join(output_dir, 'bbox')

    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(bbox_dir, exist_ok=True)

    calib_filename = '1280_720_calibration.pkl'

    # load research mode calibrations
    calibration_path = 'calibrations'

    host = None

    port_left = hl2ss.StreamPort.RM_VLC_LEFTFRONT
    calibration_lf = hl2ss_3dcv.get_calibration_rm(calibration_path, host, port_left) # i think this will read from file if it exists
    rotation_lf = hl2ss_3dcv.rm_vlc_get_rotation(port_left)
    K_left_raw, pose_left_raw = hl2ss_3dcv.rm_vlc_rotate_calibration(calibration_lf.intrinsics, calibration_lf.extrinsics, rotation_lf)
    K_left_4x4 = K_left_raw.T
    pose_left_4x4 = pose_left_raw.T
    K_left_3x3 = K_left_4x4[0:3, 0:3]

    port_right = hl2ss.StreamPort.RM_VLC_RIGHTFRONT
    calibration_rf = hl2ss_3dcv.get_calibration_rm(calibration_path, host, port_right)
    rotation_rf = hl2ss_3dcv.rm_vlc_get_rotation(port_right)
    K_right_raw, pose_right_raw = hl2ss_3dcv.rm_vlc_rotate_calibration(calibration_rf.intrinsics, calibration_rf.extrinsics, rotation_rf)
    K_right_4x4 = K_right_raw.T
    pose_right_4x4 = pose_right_raw.T
    K_right_3x3 = K_right_4x4[0:3, 0:3]


    calibrationData = None
    with open(os.path.join(calib_filename), 'rb') as f:
        calibrationData = pickle.load(f)

    print(calibrationData)

    # need to call this
    intrinsics_opencv, extrinsics_opencv = PVCalibrationToOpenCVFormat(calibrationData)

    print(intrinsics_opencv)


    model_path = os.path.join('PLY', 'trocar_fixed_joined_ascii.ply')
    ren = EfficientMeshRenderer(width=1280, height=720, camera_matrix=intrinsics_opencv, ply_model_path=model_path)
    ren_left = EfficientMeshRenderer(width=480, height=640, camera_matrix=K_left_3x3, ply_model_path=model_path)
    ren_right = EfficientMeshRenderer(width=480, height=640, camera_matrix=K_right_3x3, ply_model_path=model_path)

    # Create readers ----------------------------------------------------------
    # Stream type is detected automatically
    seq_pv = hl2ss_io.sequencer(hl2ss_io.create_rd(os.path.join(path, f'{hl2ss.get_port_name(hl2ss.StreamPort.PERSONAL_VIDEO)}.bin'), hl2ss.ChunkSize.SINGLE_TRANSFER, 'bgr24'))
    seq_vlc_left = hl2ss_io.sequencer(hl2ss_io.create_rd(os.path.join(path, f'{hl2ss.get_port_name(hl2ss.StreamPort.RM_VLC_LEFTFRONT)}.bin'), hl2ss.ChunkSize.SINGLE_TRANSFER, 'bgr24'))
    seq_vlc_right = hl2ss_io.sequencer(hl2ss_io.create_rd(os.path.join(path, f'{hl2ss.get_port_name(hl2ss.StreamPort.RM_VLC_RIGHTFRONT)}.bin'), hl2ss.ChunkSize.SINGLE_TRANSFER, 'bgr24'))
    # rd_depth = hl2ss_io.sequencer(hl2ss_io.create_rd(os.path.join(path, f'{hl2ss.get_port_name(hl2ss.StreamPort.RM_DEPTH_AHAT)}.bin'), hl2ss.ChunkSize.SINGLE_TRANSFER, 'bgr24'))

    seq_pv.open()
    rd_pv = seq_pv.get_reader()

    seq_vlc_left.open()
    seq_vlc_right.open()


    cv2.namedWindow('Video')
    cv2.namedWindow("Pose Tuner Left", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Pose Tuner Right", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Pose Tuner", cv2.WINDOW_NORMAL)



    tx=0.0
    ty=0.0
    tz=0.0
    rx=0.0
    ry=0.0
    rz=0.0

    tweak = None
    tweak_path = os.path.join(path, tweak_fname)
    if os.path.isfile(tweak_path):
        with open(tweak_path, 'rb') as f:
            tweak = pickle.load(f)
            tx, ty, tz, rx, ry, rz = tweak


    # ─── Hardcoded parameters ──────────────────────────────────────────────────────
    squaresX     = 12       # number of chessboard squares in X
    squaresY     = 7       # number of chessboard squares in Y
    squareLength = 0.068    # side length of a square (meters)
    markerLength = 0.051    # side length of the ArUco markers (meters)
    refine       = True    # whether to use refineDetectedMarkers()

    # ─── Detector setup ───────────────────────────────────────────────────────────
    detector_params = aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    dictionary_50 = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    dictionary_board = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)


    board = aruco.CharucoBoard(
        (squaresX, squaresY),
        squareLength, markerLength,
        dictionary_board
    )

    dist_coeffs = np.zeros(4)

    charuco_params = aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = refine
    charuco_params.cameraMatrix     = intrinsics_opencv
    charuco_params.distCoeffs       = dist_coeffs

    char_detector = aruco.CharucoDetector(
        board, charuco_params, detector_params
    )


    aruco_detector_50 = aruco.ArucoDetector(dictionary_50, detector_params)



    axisLength = 0.5 * min(squaresX, squaresY) * squareLength




    target_marker_id = 49

    trans_step = 0.005  # mm per keypress
    rot_step   = 2.0

    write_index = 0


    # Render decoded cont-------------------------------------------------
    while ((cv2.waitKey(1) & 0xFF) != 27):

        should_write = True

        out_left = None
        out_mask = None
        out_rgb = None
        out_right = None
        out_vis = None
        out_pose = None
        out_bbox = None


        packet = rd_pv.get_next_packet()
        if (packet is None):
            # End of file
            print('End of PV file')
            break

        _, left_vlc_packet = seq_vlc_left.get_next_packet(packet.timestamp)
        _, right_vlc_packet = seq_vlc_right.get_next_packet(packet.timestamp)

       
       
        lf_u = hl2ss_3dcv.rm_vlc_undistort(left_vlc_packet.payload.image, calibration_lf.undistort_map)
        left_image_undist_rotated = hl2ss_3dcv.rm_vlc_rotate_image(lf_u, rotation_lf)

        rf_u = hl2ss_3dcv.rm_vlc_undistort(right_vlc_packet.payload.image, calibration_rf.undistort_map)
        right_image_undist_rotated = hl2ss_3dcv.rm_vlc_rotate_image(rf_u, rotation_rf)

           
        right_image = cv2.cvtColor(right_image_undist_rotated, cv2.COLOR_GRAY2BGR)
        left_image = cv2.cvtColor(left_image_undist_rotated, cv2.COLOR_GRAY2BGR)



        cv2.imshow('Video', packet.payload.image)
        cv2.imshow('Left VLC', left_image)
        cv2.imshow('Right VLC', right_image)


        out_left = left_image_undist_rotated.copy()
        out_right = right_image_undist_rotated.copy()
        out_rgb = packet.payload.image.copy()

        
        t0 = time.perf_counter()
        

        img_masked = packet.payload.image.copy()

        corners_50, ids_50, _ = aruco_detector_50.detectMarkers(packet.payload.image)
        
        # print(ids_50)


        valid_marker_pose = False
        trocar_marker_R = None
        trocar_marker_t = None


        # find marker 49 pose. Remove it from the image for charuco detection
        if ids_50 is not None and target_marker_id in ids_50:
            print("found 49")
            # import pdb; pdb.set_trace()
            # Find the index of the target marker
            idx = np.where(ids_50 == target_marker_id)[0]
            if len(idx) > 0:
                idx = idx[0]
                # Extract its corners (shape: (1, 4, 2) for single marker)
                target_corners = corners_50[idx:idx+1]  # This is (1, 4, 2)
                fill_corners = np.round(target_corners).astype(np.int32).reshape(-1, 2)

                img_masked = cv2.fillPoly(img_masked, [fill_corners], (255,255,255))


                cv2.imshow('masked', img_masked)
                # Define 3D object points for the ArUco marker (assuming flat on z=0 plane)
                # Order: top-left, top-right, bottom-right, bottom-left
                # (counter-clockwise)
                singleMarkerLength = 36
                half_length = singleMarkerLength / 2.0
                obj_points = np.array([
                    [-half_length, half_length, 0],   # Top-left
                    [half_length, half_length, 0],    # Top-right
                    [half_length, -half_length, 0],   # Bottom-right
                    [-half_length, -half_length, 0]   # Bottom-left
                ], dtype=np.float32)
                
                # Flatten target_corners to (4, 2) for solvePnP
                img_points = target_corners[0].astype(np.float32)  # Shape: (4, 2)
                
                # Estimate pose using solvePnP
                ret, rvec, tvec = cv2.solvePnP(
                    obj_points, 
                    img_points, 
                    intrinsics_opencv, 
                    dist_coeffs
                )
                
                if ret:
                    valid_marker_pose = True
                    marker_rvec = rvec
                    trocar_marker_t = tvec
                    trocar_marker_R, _ = cv2.Rodrigues(marker_rvec)


                    R, _ = cv2.Rodrigues(rvec)

                    out_pose = make4x4(R, tvec)


        # detect Charuco
        corners, ids, markerCorners, markerIds = char_detector.detectBoard(img_masked)

        valid_pose = False
        if ids is not None and len(ids) >= 8:
            obj_pts, img_pts = board.matchImagePoints(corners, ids)
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts,
                intrinsics_opencv, dist_coeffs
            )
            valid_pose = ok

        dt = (time.perf_counter() - t0) * 1000  # ms

        vis = packet.payload.image.copy()

        
        if markerIds is not None and len(markerIds) > 0:
            aruco.drawDetectedMarkers(vis, markerCorners, markerIds)

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedCornersCharuco(vis, corners, ids, (255, 0, 0))



        if valid_marker_pose:
            cv2.drawFrameAxes(
                    vis,
                    intrinsics_opencv, dist_coeffs,
                    marker_rvec, trocar_marker_t,
                    axisLength*0.1
                )
            
        if valid_pose:
            cv2.drawFrameAxes(
                vis,
                intrinsics_opencv, dist_coeffs,
                rvec, tvec,
                axisLength
            )

            board_R, _ = cv2.Rodrigues(rvec)



            while True:
                r = scipy_R.from_euler('zyx', [rz, ry, rx], degrees=True)
                offset_R = r.as_matrix()  # shape (3,3)
                offset_t = np.array((tx,ty,tz)).reshape(-1,1)

                T_board = make4x4(board_R, tvec)


                T_board_to_marker = np.eye(4)

                T_offset = make4x4(offset_R, offset_t)

            
                
                T_trocar = T_board @ T_offset

                out_pose = T_trocar.copy()


                trocar_R, trocar_t = getRt(T_trocar)
                trocar_r, _ = cv2.Rodrigues(trocar_R)

                T_trocar_left = pose_left_4x4 @ np.linalg.inv(extrinsics_opencv) @ T_trocar
                trocar_R_left, trocar_t_left = getRt(T_trocar_left)
                
                T_trocar_right = pose_right_4x4 @ np.linalg.inv(extrinsics_opencv) @ T_trocar
                trocar_R_right, trocar_t_right = getRt(T_trocar_right)

                mask = ren.render_mask(trocar_R, trocar_t)


                mask_left = ren_left.render_mask(trocar_R_left, trocar_t_left)
                mask_right = ren_right.render_mask(trocar_R_right, trocar_t_right)

                out_mask = mask * 255
                out_bbox = mask_to_bbox_xywh(mask)

                frame = apply_mask_overlay(vis.copy(), mask, color = (0,0,255))

                bgr_left = cv2.cvtColor(left_image_undist_rotated, cv2.COLOR_GRAY2BGR)          # shape (H, W, 3)
                bgr_right = cv2.cvtColor(right_image_undist_rotated, cv2.COLOR_GRAY2BGR)          # shape (H, W, 3)
                
                bgr_left = apply_mask_overlay(bgr_left, mask_left, color = (0,0,255))
                bgr_right = apply_mask_overlay(bgr_right, mask_right, color = (0,0,255))

                


                raw_image = packet.payload.image.copy()  # Raw PV image (BGR)

                pose_data = {
                    "rotation_matrix": trocar_R.tolist(),  # 3x3 list
                    "translation": trocar_t.ravel().tolist(),  # [tx, ty, tz]
                    "rvec": trocar_r.ravel().tolist(),  # For OpenCV compatibility
                    "tvec": trocar_t.ravel().tolist()   # Same as translation
                }

                # # Save raw image
                # raw_filename = os.path.join(output_dir, 'images', f'{write_index:05d}.png')
                # cv2.imwrite(raw_filename, raw_image)

                # # Save 2D keypoints JSON
                # keypoints_filename = os.path.join(output_dir, 'annotations', f'{write_index:05d}.json')
                # with open(keypoints_filename, 'w') as f:
                #     json.dump({"keypoints": keypoints_json}, f, indent=4)

                # # Save pose JSON (separate file)
                # pose_filename = os.path.join(output_dir, 'poses', f'{write_index:05d}.json')
                # with open(pose_filename, 'w') as f:
                #     json.dump(pose_data, f, indent=4)


                # cv2.imwrite(os.path.join('/home/arssist/trocar_tracking/pose_generation/visaulization/1', f'{write_index:05d}.png'), frame)

                cv2.imshow("Pose Tuner", frame)
                cv2.imshow("Pose Tuner Left", bgr_left)
                cv2.imshow("Pose Tuner Right", bgr_right)
                out_vis = frame

                # break

                key = cv2.waitKey(0) & 0xFF

                # translation
                if key == ord('q'):    tx += trans_step
                elif key == ord('a'):    tx -= trans_step
                elif key == ord('w'):    ty += trans_step
                elif key == ord('s'):    ty -= trans_step
                elif key == ord('e'):    tz += trans_step
                elif key == ord('d'):    tz -= trans_step
                # rotation
                elif key == ord('r'):    rx += rot_step
                elif key == ord('f'):    rx -= rot_step
                elif key == ord('t'):    ry += rot_step
                elif key == ord('g'):    ry -= rot_step
                elif key == ord('y'):    rz += rot_step
                elif key == ord('h'):    rz -= rot_step
                # tweak scale
                elif key == ord('o'):
                    trans_step = 2*trans_step
                    rot_step = 2*rot_step
                elif key == ord('l'):
                    trans_step = 0.5*trans_step
                    rot_step = 0.5*rot_step
                elif key == ord('b'):
                    write_index -=1
                elif key == ord('m'):
                    should_write = False
                    break 
                elif key == 27: # esc
                    exit()
                elif key == ord(' '):
                    # write and go to next frame
                    break

                print(f"tx={tx}\nty={ty}\ntz={tz}\nrx={rx}\nry={ry}\nrz={rz}")
                print(f"next write index={write_index}")

                with open(tweak_path, 'wb') as f:
                    pickle.dump((tx, ty, tz, rx, ry, rz), f)
                    # print("saving to ", tweak_path)

            if should_write:
                print(f"writing with index={write_index}")

                # save all output here
                cv2.imwrite(os.path.join(left_dir, f'{write_index:05d}.png'), out_left)
                cv2.imwrite(os.path.join(mask_dir, f'{write_index:05d}.png'), out_mask)
                cv2.imwrite(os.path.join(rgb_dir, f'{write_index:05d}.png'), out_rgb) 
                cv2.imwrite(os.path.join(right_dir, f'{write_index:05d}.png'), out_right)
                cv2.imwrite(os.path.join(vis_dir, f'{write_index:05d}.png'), out_vis)

                np.savetxt(os.path.join(pose_dir, f'{write_index:05d}.txt'), out_pose)
                np.savetxt(os.path.join(bbox_dir, f'{write_index:05d}.txt'), out_bbox)

                write_index += 1


cv2.destroyAllWindows()
