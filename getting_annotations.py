import os

def centerpoint(bboxtopleftx, bboxtoplefty, width, height):

    #calculate the center point of a bounding box
    x_center = bboxtopleftx + (width / 2)
    y_center = bboxtoplefty + (height / 2)

    return x_center, y_center

def main():
    
    annotation_folder = "original_annotations"
    # creating an output folder for all of the annotations 
    output_folder = 'annotations_for_frames'

    # if there is no folder currently for the output annotations create one
    if not os.path.exists(output_folder):
        
        # create a folder based on the path we described
        os.makedirs(output_folder)

    else:
        # return if the folder has already been created
        pass
    
    # go through every file within original_annotations folder
    for filename in os.listdir(annotation_folder):

        # if the filename has .txt extension star the code
        if filename.lower().endswith((".txt")):

            #path to the annotation file
            annotations_path = os.path.join(annotation_folder, filename)

            #extracts base name w/o extension for naming the output
            annotation_name = os.path.splitext(filename)[0]

            # create a empty list titled lines to save the lines we will read in from the annotations
            values = []
            
            # we know that VIRAT.txt file is in the data format of object id, object duration, current frame, bboxleft topx, bbox lefttopy, bboxwidth, bbox height, object type
            # so from this we know that from the data we only need to keep 2, 3, 4, 5, 6, 7 values from the lines
            with open(annotations_path, 'r') as file:

                # reads in all of the current lines in the .txt file and saves them as a list of strings
                lines = file.readlines()

                # iterate over all of the lines within the .txt file
                for line in lines:
                    
                    # Remove extra whitespace and split by space
                    split_lines = line.strip().split(' ')

                    # creates
                    temp = []
                    
                    for i in range(len(split_lines)):
                        
                        # we do not need the values at 0, 1 for yolo format so if we hit those values continue
                        if (i == 0 or i == 1):
                        
                            continue
                        # else we continue to append the values we need to utilize
                        else:

                            # append the values needed for yolo format into the temp variable
                            temp.append(int(split_lines[i]))
                    
                    # append the values into the values list as now we have the data in row format
                    values.append(temp)
            
            # create an empty list for the annotaions
            annotations = []
            
            # create an empty list to store frames 
            biggest_frame = 0

            # iterate over each row found within the annotation file
            for i in range(len(values)):
                
                # create a temporary list to store values in rows
                temporary_list = []

                # this give us the class id
                temporary_list.append(values[i][5])

                # function that finds the x_center and y_center for yolo format
                x_center, y_center = centerpoint(values[i][1], values[i][2], values[i][3], values[i][4])
                
                # append the x_center, y_center values, width of the bounding box, height of the bounding box, and the frames to the temporary list
                temporary_list.append(x_center)
                temporary_list.append(y_center)
                temporary_list.append(values[i][3])
                temporary_list.append(values[i][4])
                temporary_list.append(values[i][0])

                # append the temporaray list to the annoations list to create rows of values
                annotations.append(temporary_list)

                #Updates biggest_frame if current frame is larger
                if values[i][0] > biggest_frame:

                    biggest_frame = values[i][0]
                    
            for i in range(biggest_frame + 1):
                #create separate annotation file for each frame
                annotations_filename = os.path.join(output_folder, f"{annotation_name}_frame_{i}.txt")

                for j in range(len(annotations)):
                    
                    # if the annoation at j is equal to the frame we must then write to the file the information at annotations[j][5]
                    if annotations[j][5] == i:

                        with open(annotations_filename, "a") as file:

                            # create a copy of the annotations at j excluding the frames because yolo does not need it
                            anno_copy = annotations[j][:5]
                            # conver the list of integers to string then write it to the file
                            file.write(" ".join(map(str, anno_copy)) + "\n")
                            
if __name__ == '__main__':

    main()