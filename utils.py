import numpy as np
# The PV calibration data is strange... I looked into the hl2ss/viewer examples as well as functions like
# hl2ss_3dcv.pv_fix_calibration to figure it out. 
def PVCalibrationToOpenCVFormat(hl2ss_calibration):
    fx, fy, = hl2ss_calibration.focal_length
    cx, cy = hl2ss_calibration.principal_point
    
    intrinsics_opencv = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,   1]
    ])
    
    R = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=hl2ss_calibration.extrinsics.dtype)

    extrinsics = hl2ss_calibration.extrinsics @ R
    extrinsics_opencv = extrinsics.T

    return intrinsics_opencv, extrinsics_opencv

def printHl2ssCalibration(calibrationData):
    print('================Calibration================')
    print(f'Focal length: {calibrationData.focal_length}')
    print(f'Principal point: {calibrationData.principal_point}')
    print(f'Radial distortion: {calibrationData.radial_distortion}')
    print(f'Tangential distortion: {calibrationData.tangential_distortion}')
    print('\nProjection')
    print(calibrationData.projection)
    print('\nIntrinsics')
    print(calibrationData.intrinsics)
    print('\nRigNode Extrinsics')
    print(calibrationData.extrinsics)
    print(f'\nIntrinsics MF: {calibrationData.intrinsics_mf}')
    print(f'Extrinsics MF: {calibrationData.extrinsics_mf}')
    print("")