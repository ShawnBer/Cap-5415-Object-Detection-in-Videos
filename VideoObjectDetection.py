from ultralytics import YOLO
import cv2
import os

def main():

    # sets the model to our pretrained model that we trained earlier
    model = YOLO('runs/detect/Fifth_Train/weights/best.pt')

    # get the video path we want to deploy our model on
    video_path = 'Videos/Original/VIRAT_S_050200_00_000106_000380.mp4'
    
    # setting up video output destination
    output_path = 'Videos/Model_Output/Fourth_Model_Video5.mp4'
    os.makedirs(os.path.dirname(output_path), exist_ok = True)

    # begin video capture for open cv
    cap = cv2.VideoCapture(video_path)

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