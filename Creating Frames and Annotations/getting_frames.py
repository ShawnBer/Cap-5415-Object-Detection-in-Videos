import cv2 
import os

def main():
    
    #Folder with only videos
    video_folder = "input_videos"
    output_folder = "frames_output"
    
    #create base output folder
    os.makedirs(output_folder, exist_ok=True)

    #intialzize the frame counter
    frame_count = 0

    for filename in os.listdir(video_folder):
        if filename.lower().endswith((".mp4")):
            video_path = os.path.join(video_folder, filename)

            #extracts base name w/o extension for naming the output
            video_name = os.path.splitext(filename)[0]

            #opens video
            cap = cv2.VideoCapture(video_path)
            #set the frame count to zero before reading in frames

            #if not cap.isOpened():
            #print('Error: no open video file.')
            #i = 0
            #while a video capture is open loop over all of the frames
            while cap.isOpened():
                
                #this returns if it was able to read in a frame (ret), and it also gets the actual frame at each point
                ret, frame = cap.read()

                #if it is not able to find a image/frame we break the loop
                if not ret:
                    
                    #end of video
                    break

                #if frame_count == 23:
                #i += 1
                #process the frame
                print(f"Processing frame {frame_count}")
                #gray_frame = cv2.cvtcolor(frame, cv2.COLOR_BGR2GRAY)
                frame_filename = os.path.join(output_folder, f"{frame_count}.jpg")
                cv2.imwrite(frame_filename, frame)
                #frame_count = 0
                
                frame_count += 1
            
            #here we close the video file
            cap.release()
            print(f"saved {frame_count} frames to '{output_folder}'")
    
if __name__ == '__main__':

    main()