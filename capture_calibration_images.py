import cv2
import os
import time

# Capture parameters
CAMERA_ID = 0  # Camera ID (usually 0 for built-in webcam)
CHESSBOARD_SIZE = (9, 6)  # Number of inner corners per chessboard row and column
OUTPUT_DIRECTORY = 'calibration_images'  # Directory to save calibration images

IMAGE_RES = (1280,720)



def capture_calibration_images():
    """
    Capture images of a chessboard pattern for camera calibration.
    Run in an infinite loop until user presses 'q' or Escape to quit.
    Press 'c' to capture an image.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    
    # Open camera
    cap = cv2.VideoCapture(CAMERA_ID)

    # Set width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_RES[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_RES[1])
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_ID}")
        return
    
    # Get camera resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")
    
    # Counter for captured images
    img_counter = 0
    
    print("Press 'c' to capture an image")
    print("Press 'q' or Escape to quit")
    print(f"Images will be saved to {OUTPUT_DIRECTORY}")
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        og_frame = frame.copy() # save original image to calibrate
        
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Convert to grayscale for chessboard detection

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret_chess, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        
        # Draw corners if found
        if ret_chess:
            # Draw and display the corners
            cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, ret_chess)
            cv2.putText(frame, "Chessboard detected!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display capture counter
        cv2.putText(frame, f"Captured: {img_counter}", (50, height - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Camera Calibration', frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' or Escape to quit
        if key == ord('q') or key == 27:  # 27 is the ASCII code for Escape
            print("Exiting...")
            break
        
        # 'c' to capture
        elif key == ord('c'):
            # Save the image
            img_name = os.path.join(OUTPUT_DIRECTORY, f"calibration_{img_counter:02d}.jpg")
            cv2.imwrite(img_name, og_frame)
            print(f"Captured {img_name}")
            
            img_counter += 1
    
    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Captured {img_counter} images for calibration")

if __name__ == "__main__":
    capture_calibration_images()
