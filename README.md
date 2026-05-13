# Sports Form Pose Extraction with MediaPipe

This project uses MediaPipe Pose to extract body keypoints from sports videos and generate movement curves for correct vs. incorrect form comparison.

Current example:
- Sport: Badminton
- Form: Overhead clear
- Comparison: Correct form vs. incorrect form

## Files

- `extract_pose.py`  
  Extracts pose keypoints from two videos and saves the results as CSV files.

- `plot_pose.py`  
  Reads the CSV files and generates curve plots.

- `requirements.txt`  
  Contains the Python packages needed to run the project.

## Input Videos

Place two videos in the project folder:

```text
correct_form.mp4
incorrect_form.mp4
