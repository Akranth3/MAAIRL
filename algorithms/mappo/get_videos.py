import numpy as np
import cv2
from matplotlib import pyplot as plt

def create_video_from_numpy(numpy_array, output_filename='output.mp4', fps=30):
    """
    Convert a numpy array of shape (frames, height, width, channels) to a video file.
    
    Parameters:
    -----------
    numpy_array : np.ndarray
        4D array of shape (frames, height, width, channels)
    output_filename : str
        Name of the output video file (should end in .mp4)
    fps : int
        Frames per second for the output video
    """
    # Get dimensions
    n_frames, height, width, channels = numpy_array.shape
    
    # Create video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    
    # Write each frame
    for i in range(n_frames):
        # Convert from RGB to BGR (OpenCV uses BGR)
        frame = cv2.cvtColor(numpy_array[i], cv2.COLOR_RGB2BGR)
        
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
            
        # Write the frame
        out.write(frame)
        
        # Optional: show progress
        if i % 10 == 0:
            print(f"Processing frame {i}/{n_frames}")
    
    # Release the video writer
    out.release()
    print(f"Video saved as {output_filename}")

# Load and process your data
location = "videos/episode_1.npy"
data = np.load(location)

# Create the video
create_video_from_numpy(data, output_filename='episode_0.mp4', fps=30)

# # Optional: Display first frame to verify
# plt.imshow(data[0])
# plt.axis('off')
# plt.show()