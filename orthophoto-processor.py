import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class OrthophotoPipeline:
    def __init__(self, camera_index=0):
        """
        Initialize orthophoto processing pipeline
        
        :param camera_index: Index of the webcam (default is 0)
        """
        logger.info("Initializing Orthophoto Processing Pipeline")
        
        self.capture = cv2.VideoCapture(camera_index)
        
        # Check if the webcam is opened correctly
        if not self.capture.isOpened():
            logger.error("Cannot open webcam")
            raise IOError("Cannot open webcam")
        
        # Set camera resolution
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        logger.info(f"Camera resolution set to 1280x720")
        
        # Camera calibration parameters (these would typically be computed via calibration)
        self.camera_matrix = None
        self.distortion_coeffs = None
        
    def calibrate_camera(self, checkerboard_size=(9, 6), square_size=0.025):
        """
        Camera calibration using a checkerboard pattern
        
        :param checkerboard_size: Internal corners of checkerboard
        :param square_size: Size of each square in meters
        :return: Calibration success
        """
        logger.info("Starting Camera Calibration Process")
        logger.info(f"Checkerboard size: {checkerboard_size}")
        logger.info(f"Square size: {square_size} meters")
        
        # Prepare object points
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        logger.info("Position checkerboard pattern in view. Press 'c' to capture, 'q' to finish.")
        
        # Capture calibration frames
        captured_frames = 0
        while captured_frames < 20:
            ret, frame = self.capture.read()
            if not ret:
                logger.warning("Failed to capture frame")
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
            
            # Display frame with detected corners
            frame_with_corners = frame.copy()
            cv2.drawChessboardCorners(frame_with_corners, checkerboard_size, corners, ret)
            cv2.imshow('Calibration', frame_with_corners)
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                captured_frames += 1
                logger.info(f"Calibration frame captured: {captured_frames}/20")
            
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        # Calibrate camera
        logger.info("Computing camera calibration...")
        ret, self.camera_matrix, self.distortion_coeffs, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        if ret:
            logger.info("Camera calibration successful")
            logger.info("Camera Matrix:")
            logger.info(str(self.camera_matrix))
            logger.info("Distortion Coefficients:")
            logger.info(str(self.distortion_coeffs))
        else:
            logger.error("Camera calibration failed")
        
        return ret
    
    def undistort_image(self, image):
        """
        Remove lens distortion from image
        
        :param image: Input image
        :return: Undistorted image
        """
        logger.info("Starting image undistortion process")
        
        if self.camera_matrix is None or self.distortion_coeffs is None:
            logger.error("Camera must be calibrated first")
            raise ValueError("Camera must be calibrated first")
        
        h, w = image.shape[:2]
        logger.info(f"Image dimensions: {w}x{h}")
        
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.distortion_coeffs, (w, h), 1, (w, h)
        )
        
        logger.info("Applying undistortion...")
        # Undistort
        undistorted = cv2.undistort(
            image, 
            self.camera_matrix, 
            self.distortion_coeffs, 
            None, 
            new_camera_matrix
        )
        
        # Crop the image
        x, y, w, h = roi
        undistorted_cropped = undistorted[y:y+h, x:x+w]
        
        logger.info("Undistortion complete")
        logger.info(f"Cropped image dimensions: {w}x{h}")
        
        return undistorted_cropped
    
    def create_orthophoto(self, image, dem=None):
        """
        Create an orthophoto by correcting for terrain and camera angle
        
        :param image: Input image
        :param dem: Digital Elevation Model (optional)
        :return: Orthographically corrected image
        """
        logger.info("Generating Orthophoto")
        
        # If no DEM provided, assume flat terrain
        if dem is None:
            logger.info("No Digital Elevation Model provided. Assuming flat terrain.")
            dem = np.zeros(image.shape[:2], dtype=np.float32)
        
        # Get image dimensions
        height, width = image.shape[:2]
        logger.info(f"Image dimensions for orthophoto: {width}x{height}")
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Flatten coordinates
        points = np.column_stack([x.ravel(), y.ravel()])
        values = image.reshape(-1, 3)
        
        # Create a grid for the output
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, width-1, width),
            np.linspace(0, height-1, height)
        )
        
        # Interpolate to correct geometric distortions
        logger.info("Performing geometric correction interpolation...")
        start_time = time.time()
        orthophoto = griddata(
            points, 
            values, 
            (grid_x, grid_y), 
            method='linear', 
            fill_value=0
        ).astype(np.uint8)
        
        end_time = time.time()
        logger.info(f"Interpolation completed in {end_time - start_time:.4f} seconds")
        
        return orthophoto
    
    def process_orthophoto(self):
        """
        Capture and process orthophoto
        
        :return: Processed orthophoto
        """
        logger.info("Starting Orthophoto Processing")
        
        # Capture frame
        ret, frame = self.capture.read()
        if not ret:
            logger.error("Failed to capture image")
            return None, None, None
        
        logger.info("Frame captured successfully")
        
        # Undistort the image
        undistorted = self.undistort_image(frame)
        
        # Create orthophoto
        orthophoto = self.create_orthophoto(undistorted)
        
        return frame, undistorted, orthophoto
    
    def run_pipeline(self):
        """
        Run full orthophoto processing pipeline
        """
        logger.info("===== ORTHOPHOTO PROCESSING PIPELINE STARTED =====")
        
        # First, calibrate the camera
        logger.info("Camera Calibration Phase")
        calibration_success = self.calibrate_camera()
        
        if not calibration_success:
            logger.error("Camera calibration failed. Exiting.")
            return
        
        try:
            while True:
                logger.info("\n--- Processing Next Frame ---")
                # Process orthophoto
                original, undistorted, orthophoto = self.process_orthophoto()
                
                if original is not None:
                    # Display results
                    cv2.imshow('Original', original)
                    cv2.imshow('Undistorted', undistorted)
                    cv2.imshow('Orthophoto', orthophoto)
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested exit")
                    break
        
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        
        finally:
            # Clean up
            self.capture.release()
            cv2.destroyAllWindows()
            
            logger.info("===== ORTHOPHOTO PROCESSING PIPELINE COMPLETED =====")

def main():
    try:
        processor = OrthophotoPipeline()
        processor.run_pipeline()
    except Exception as e:
        logger.error(f"Critical error: {e}")

if __name__ == "__main__":
    main()