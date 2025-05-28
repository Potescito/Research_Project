# %%
import cv2
import os
import sys

print("--- Terminal Video Test ---")
print(f"Python Executable: {sys.executable}")
print(f"OpenCV Version: {cv2.__version__}")
print(f"OpenCV Path: {cv2.__file__}")
print(f"Current Working Directory: {os.getcwd()}")

# You can try uncommenting these to see if they influence backend choice
# os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"
# os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"

video_path = r"C:\Users\juli-\Downloads\Research_Project\data\dataset_2drt_video_only\sub073\2drt\video\sub073_2drt_04_bvt_r2_video.mp4" # <--- IMPORTANT: Use an ABSOLUTE path to a known good video file

print(f"\nAttempting to open: {video_path}")

print("\nOpenCV Build Information (relevant parts):")
build_info = cv2.getBuildInformation()
for line in build_info.splitlines():
    if "Video I/O" in line:
        print(line)
    if "FFMPEG" in line:
        print(line)
    if "GStreamer" in line:
        print(line)

print("\nAvailable VideoIO Backends:")
try:
    backends = cv2.videoio.getStreamBackends()
    for backend in backends:
        print(f"  {backend.name()} - {backend.id()}")
except AttributeError:
    print("  cv2.videoio.getStreamBackends() not available in this OpenCV version.")


print("\nAttempting with default backend:")
cap_default = cv2.VideoCapture(video_path)
if cap_default.isOpened():
    print("  Successfully opened with default backend.")
    ret, _ = cap_default.read()
    print(f"  Read frame successful: {ret}")
    backend_name_default = cap_default.getBackendName() # Requires OpenCV 4.2+
    print(f"  Actual backend used (default): {backend_name_default}")
    cap_default.release()
else:
    print("  Failed to open with default backend.")

print("\nAttempting with FFmpeg explicitly (cv2.CAP_FFMPEG):")
cap_ffmpeg = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
if cap_ffmpeg.isOpened():
    print("  Successfully opened with FFmpeg backend.")
    ret, _ = cap_ffmpeg.read()
    print(f"  Read frame successful: {ret}")
    backend_name_ffmpeg = cap_ffmpeg.getBackendName() # Requires OpenCV 4.2+
    print(f"  Actual backend used (FFmpeg): {backend_name_ffmpeg}")
    cap_ffmpeg.release()
else:
    print("  Failed to open with FFmpeg backend.")

print("\nAttempting with GStreamer explicitly (cv2.CAP_GSTREAMER):")
cap_gstreamer = cv2.VideoCapture(video_path, cv2.CAP_GSTREAMER)
if cap_gstreamer.isOpened():
    print("  Successfully opened with GStreamer backend.")
    ret, _ = cap_gstreamer.read()
    print(f"  Read frame successful: {ret}")
    backend_name_gstreamer = cap_gstreamer.getBackendName() # Requires OpenCV 4.2+
    print(f"  Actual backend used (GStreamer): {backend_name_gstreamer}")
    cap_gstreamer.release()
else:
    print("  Failed to open with GStreamer backend.")

print("\n--- Test Complete ---")
# %%
