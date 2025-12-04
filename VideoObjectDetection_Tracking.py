from ultralytics import YOLO
import numpy as np
import cv2
import os
from pathlib import Path

def get_count(yolo_class_ids, model_names):

    # initializing variables for tracking class count and converting the tensor we get to a list
    classid = 0
    count = 0
    classcount = []
    listofclassids = yolo_class_ids.tolist()
    i = 0

    # keep iterating over until we get a count of all of the class ids
    while classid < len(model_names):
        
        # this utilizes .count() function to add up all occurences of i in the list
        count = listofclassids.count(i)

        # append to the count list how many classes we found at i
        classcount.append(count)

        # incrementing values needed and setting count back to zero
        classid += 1   
        count = 0
        i += 1

    return classcount

def get_centroid(bounding_boxes):

    # creating a empty list to store the centroids and initializing a temp variable i
    centroids= []
    i = 0

    # iterate over all of the boxes within the bounding boxes that yolo gives
    for box in bounding_boxes:

        # convert the boundingboxes.xyxy data to a list and then set x1, y1, x2, y2 equal to it 
        x1, y1, x2, y2 = bounding_boxes.xyxy[i].tolist()
        
        # centroidx = (xtop_left + xbottom_right)/2
        # centroidy = (ytop_left + ybottom_right)/2
        # calculate the centroids of the bounding boxes by utilizing the formula above
        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2

        # append the centroid to the list and convert it to integer
        centroid_x_y = (int(centroid_x), int(centroid_y))
        centroids.append(centroid_x_y)
        i += 1
    
    return centroids

def get_centroid_euclidian(current_centroids, old_centroids):

    # convert current centroids and past centroid into a numpy array for calculating euclidian distance
    current_centroids = np.array(current_centroids)
    old_centroids = np.array(old_centroids)

    # initialize list for the final centroids and the id list
    final_centroids = []
    id_list = []

    # iterate over the current centroids of the frame
    for i in range(len(current_centroids)): 

        # reset variable ever iteration to make sure we don't keep previous iteration values
        lowest = 9999999
        tempj = 0

        # iterate over the old centroids we have already found
        for j in range(len(old_centroids)):
            
            # calculate the distance between the current and the old centroids to find if there are any that are similiar
            euclidian_distance = np.sqrt(np.sum((current_centroids[i] - old_centroids[j]) ** 2))

            # if j has not already been assigned as an id to something check to see if the euclidian distance is lower than a threshold and if it is assign the id of j to it
            # if j not in id_list:
                
            # check to see if it is the lowest distance found so far
            if euclidian_distance < lowest:
                
                # if it is set lowest to this new distance and store the iteration it happend for the id
                lowest = euclidian_distance
                tempj = j

        # add the tracking id we found to the list of ids and append the centroids
        id_list.append(tempj)
        final_centroids.append(current_centroids[i])
                    
    return final_centroids, id_list

def main():

    script_dir = Path(__file__).resolve().parent

    # sets the model to our pretrained model that we trained earlier
    model_path = (script_dir / "runs" / "detect" / "Sixth_Train" / "weights" / "best.pt")

    # get the video path we want to deploy our model on
    # important to set the video path of the video you want to test here
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
    
    
    # Load YOLO model
    model = YOLO(str(model_path))



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

    # initializing old_centroids
    old_centroids = []

    # while a video capture is open loop over all of the frames
    while cap.isOpened():
        
        # this returns if it was able to read in a frame (ret), and it also gets the actual frame at each point
        ret, frame = cap.read()

        # if it is not able to find a image/frame we break the loop
        if not ret:
                    
            #end of video
            break
        
        # have the trained model make an inference on the frame
        results = model(frame, verbose = False, conf = 0.4, iou = 0.4)
        
        # this gives us the detection information for information relating to the bounding boxes (coordinates, confidence, and class ids)
        boxes = results[0].boxes

        # this gives us a tensor of all of the predictions classes so for ex: tensor([1., 1., 2., 0., ...])
        current_classes = boxes.cls

        # plots the annotated results at each inference to a new frame
        bounding_box_frame = results[0].plot()
        
        # get the number of inferences for each class
        # model.names returns a dict of all of the class names so for our case ex: {0: 'Bike/Bicycle', 1: 'Car', 2: 'Person'}
        # calulate the centroids of all of the bounding boxes from my function
        class_count = get_count(current_classes, model.names)
        centroids = get_centroid(boxes)

        # check to see if we have centroid values stored
        if old_centroids:
            
            # get the euclidian distance of the new centroids and old centroids
            compared_centroids, id_list = get_centroid_euclidian(centroids, old_centroids)

            # initialize j as zero to iterate over the centroids to put them on the image
            j = 0

            # iterate over the amonut of centroids found and display them utilizing cv2.cirle
            while j < len(compared_centroids):
                
                # put on the output frame the circle for the centroid of the object
                # cv2.circle(bounding_box_frame, compared_centroids[j], 5, (255, 0, 0), -1)

                # display the id that is associated with the centroids between frames and increment j
                cv2.putText(bounding_box_frame, f'id: {id_list[j]}', compared_centroids[j], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 125))
                j += 1
            
        # initialize i as zero to iterate over the model names for putting text on the image
        i = 0
        # set values for the rgb values for the text for each class
        rgb_values = [(255, 0, 0), (255, 255, 0), (255, 255, 255)]
        # set positions for the text 
        positions = [(10, 30), (10, 60), (10, 90)]

        # iterate over the amonut of names setup in the model and create a visual place for the class
        while i < len(model.names):

            # put on the output frame the number of classes for each inference and give each class a color
            cv2.putText(bounding_box_frame, f'{model.names[i]}: {class_count[i]}', positions[i], cv2.FONT_HERSHEY_SIMPLEX, 1, rgb_values[i])
            i += 1

        # shows the frame that the model did an inference on
        cv2.imshow('Yolov11 Inferences', bounding_box_frame)

        # write the frame to save to video for later review
        video_out.write(bounding_box_frame)

        # store this iterations centroids as old_centroids
        old_centroids = centroids

        # if you press q it will break the loop early
        if cv2.waitKey(1) & 0xFF == ord('q'):

            break
   
    print(f'Video Finished, output saved to: {output_path}')
    cap.release()
    video_out.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    
    main()