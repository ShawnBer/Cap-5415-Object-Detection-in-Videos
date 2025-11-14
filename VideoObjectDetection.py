from ultralytics import YOLO
import cv2
import os

def main():

    # sets the model to our pretrained model that we trained earlier
    model = YOLO("runs/detect/train2/weights/best.pt")

    # get the video path we want to deploy our model on
    video_path = "Videos/Original/VIRAT_S_000201_01_000384_000589.mp4"

    # setting up video output destination
    output_path = "Videos/Model_Output/Output.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok = True)

    # beginn video capture for open cv
    cap = cv2.VideoCapture(video_path)

    # get video information for the cv2.VideoWrite function
    framewidth = int(cap.get(3))
    frameheight = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'Video width: {framewidth}, Video height: {frameheight}, FPS: {fps}')

    # setting up writing to video for display after done running
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    video_out = cv2.VideoWriter(output_path, fourcc, fps, (framewidth, frameheight))

    # while a video capture is open loop over all of the frames
    while cap.isOpened():
                
        # this returns if it was able to read in a frame (ret), and it also gets the actual frame at each point
        ret, frame = cap.read()

        # if it is not able to find a image/frame we break the loop
        if not ret:
                    
            #end of video
            break
        
        # have the pretrained model make an inference of the frame
        results = model(frame)
        
        # plots the annotated results at each inference
        annotated_result = results[0].plot()

        # shows the frame that the model did an inference on
        cv2.imshow('Yolov11 Inference', annotated_result)

        # write the frame to save to video for later review
        video_out.write(annotated_result)

        # if you press q it will break the loop early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
   
    cap.release()
    video_out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
