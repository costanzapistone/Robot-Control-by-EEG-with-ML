import numpy as np
import cv2
import os
from pathlib import Path

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}
# Number of markers to generate
num_markers = 3

# Dictionary to use
aruco_type = "DICT_4X4_50"

# Create the aruco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

# Marker size and ID
marker_size = 300

# Specify the output folder relative to the script
script_directory = Path(__file__).resolve().parent
output_folder = script_directory / "aruco_markers"

# Check if the folder already exists
if not output_folder.exists():
    output_folder.mkdir()

# Check if the folder already exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Generate and save multiple markers
for marker_id in range(1, num_markers + 1):
    # Generate a single marker
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

    # Save the marker image using os.path.join
    marker_filename = os.path.join(output_folder, f"marker_{marker_id}.png")
    cv2.imwrite(marker_filename, marker_img)

    print(f"Generated and saved ArUCo marker with ID {marker_id} to {marker_filename}")

