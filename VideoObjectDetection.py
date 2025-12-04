from ultralytics import YOLO
import cv2
import os
from pathlib import Path

def main():

    script_dir = Path(__file__).resolve().parent

    # sets the model to our pretrained model that we trained earlier
    model_path = script_dir / "runs" / "detect" / "Sixth_Train" / "weights" / "best.pt"
    model = YOLO(str(model_path))

    # get the video path we want to deploy our model on
    video_path = script_dir / "Videos" / "Original" / "VIRAT_S_010001_09_000921_000952.mp4"
    
    # setting up video output destination
    output_path = script_dir / "Videos" / "Model_Output" / "output.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"Error: model weights not found: {model_path}")
        return
    if not video_path.exists():
        print(f"Error: input video not found: {video_path}")
        return

    # begin video capture for open cv
    cap = cv2.VideoCapture(str(video_path))

    # get video information for the cv2.VideoWrite function
    framewidth = int(cap.get(3))
    frameheight = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'Video width: {framewidth}, Video height: {frameheight}, FPS: {fps}')

    # setting up writing to video for display after done running
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_out = cv2.VideoWriter(output_path, fourcc, fps, (framewidth, frameheight))

    # while a video capture is open loop over all of the frames
    while cap.isOpened():
                
        # this returns if it was able to read in a frame (ret), and it also gets the actual frame at each point
        ret, frame = cap.read()

        # if it is not able to find a image/frame we break the loop
        if not ret:
                    
            #end of video
            break
        
        # have the trained model make an inference on the frame
        results = model(frame, verbose = False)

        # plots the annotated results at each inference
        bounding_box_frame = results[0].plot()

        # shows the frame that the model did an inference on
        cv2.imshow('Yolov11 Inferences', bounding_box_frame)

        # write the frame to save to video for later review
        video_out.write(bounding_box_frame)

        # if you press q it will break the loop early
        if cv2.waitKey(1) & 0xFF == ord('q'):

            break
    
    print(f'Video Finished, output saved to: {output_path}')
    cap.release()
    video_out.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    
    main()