import pandas as pd
import matplotlib.pyplot as plt


correct = pd.read_csv("correct_pose.csv")
incorrect = pd.read_csv("incorrect_pose.csv")

print("Correct rows:", len(correct))
print("Incorrect rows:", len(incorrect))

correct = correct[correct["detected"] == 1]
incorrect = incorrect[incorrect["detected"] == 1]

print("Correct detected rows:", len(correct))
print("Incorrect detected rows:", len(incorrect))

if len(correct) == 0 or len(incorrect) == 0:
    print("No pose detected in one of the videos. Try a clearer video.")
    exit()


# 1. Right elbow height over time
plt.figure()
plt.plot(correct["frame"], correct["right_elbow_y"], label="Correct")
plt.plot(incorrect["frame"], incorrect["right_elbow_y"], label="Incorrect")
plt.xlabel("Frame")
plt.ylabel("Right elbow y-position")
plt.title("Right Elbow Height Over Time")
plt.legend()
plt.gca().invert_yaxis()
plt.savefig("elbow_height.png", dpi=300)
plt.show()


# 2. Right elbow angle over time
plt.figure()
plt.plot(correct["frame"], correct["right_elbow_angle"], label="Correct")
plt.plot(incorrect["frame"], incorrect["right_elbow_angle"], label="Incorrect")
plt.xlabel("Frame")
plt.ylabel("Right elbow angle degree")
plt.title("Right Elbow Angle Over Time")
plt.legend()
plt.savefig("elbow_angle.png", dpi=300)
plt.show()


# 3. Right wrist trajectory
plt.figure()
plt.plot(correct["right_wrist_x"], correct["right_wrist_y"], label="Correct")
plt.plot(incorrect["right_wrist_x"], incorrect["right_wrist_y"], label="Incorrect")
plt.xlabel("Right wrist x-position")
plt.ylabel("Right wrist y-position")
plt.title("Right Wrist Trajectory")
plt.legend()
plt.gca().invert_yaxis()
plt.savefig("wrist_trajectory.png", dpi=300)
plt.show()