import cv2
import numpy as np
from sympy import flatten

cam = cv2.VideoCapture(0)

marker_length = 0.03

while True:
    ret, frame = cam.read()
    cv2.imshow("frame", frame)

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    corners, ids, _ = detector.detectMarkers(frame)

    camera_matrix = np.array([
        [576, 0, 320],
        [0, 576, 240],
        [0, 0, 1]
    ], dtype=np.float32)

    objps = np.array([
        [-marker_length / 2, marker_length / 2, 0],
        [marker_length / 2, marker_length / 2, 0],
        [marker_length / 2, -marker_length / 2, 0],
        [-marker_length / 2, -marker_length / 2, 0]
    ])

    for cs in corners:
        img_pts = cs.reshape(4, 2).astype(np.float32)

        success, rvec, tvec = cv2.solvePnP(
            objps.astype(np.float32),
            img_pts,
            camera_matrix,
            np.load("/home/mik/SchoolProjects/NTO_TOURS/noetic/dist_coeffs.npy"),
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        print(success, tvec.flatten())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
