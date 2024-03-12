import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the video
cap = cv2.VideoCapture('Dataset\Footage1.mp4')

# Function to process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerHSV = [10, 140, 170]
    UpperHSV = [25, 255, 255]


    mask = cv2.inRange(hsv, np.array(lowerHSV), np.array(UpperHSV))

    DT = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

    Object = np.where(DT > DT.max() * 0.4, 1, 0)

    BG = np.where(DT == 0, 1, 0)
    corner = ((Object + BG) != 0).astype(np.uint8)

    n, lek = cv2.connectedComponents(corner)

    lek += 2
    lek[BG == 1] = 1
    lek[corner == 0] = -1
    for i, n in enumerate(np.unique(lek)[1:], 1):
        lek[lek == n] = i

    lek = cv2.watershed(frame, lek)

    contour = []
    for i in range(2,lek.max()+1): #เริ่มที่ 2 เพราะเราไม่เอา Background
        c,_ = cv2.findContours((lek==i).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if c:  # Check if any contours were found
            contour.append(c[0])

    areas = []
    for i in range(len(contour)):
        a = cv2.contourArea(contour[i])
        areas.append(a)
        # print("index:", i, "area:", a)
        x,y,w,h = cv2.boundingRect(contour[i])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),thickness=2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(frame.shape[0], frame.shape[1]) / 400.0  # Adjusting font size
    text_position = (int(frame.shape[1] * 0.01), int(frame.shape[0] * 0.07)) # Adjusting font position
    cv2.putText(frame, f'Number of top layer oranges: {len(contour)}', text_position, font, font_scale, (0, 255, 0), 2)
    plt.imshow(frame)

    # Calculate mean and standard deviation of contour areas
    mean_area = np.mean(areas)
    std_area = np.std(areas)

    # Define thresholds based on mean and standard deviation
    below_threshold = mean_area -  std_area
    above_threshold = mean_area +  std_area

    # Classify areas into small, medium, and large
    small_areas = [area for area in areas if area <= below_threshold]
    medium_areas = [area for area in areas if below_threshold <= area <= above_threshold]
    large_areas = [area for area in areas if area > above_threshold]


    #average diameters for oranges ranging from 2.29 inches(5.8166cm) to 3.47 inches(8.8138cm), the avereage's average would be 7.3152cm
    #Calculating number of oranges using volume_basket/volume_orange would not be pratical cus of the space between the orange. So,I use layer instead(assuming the basket was rectangular or cylindrical)
    height = 0 #assuming height in centimeter
    diameter = 7.3152 #average orange's diameters 
    layer = int(height//diameter) #we don't want any decimal
    if (layer < 1):
        layer = 1

    S_text_position = (int(frame.shape[1] * 0.01), int(frame.shape[0] * 0.2))
    M_text_position = (int(frame.shape[1] * 0.01), int(frame.shape[0] * 0.3))
    L_text_position = (int(frame.shape[1] * 0.01), int(frame.shape[0] * 0.4))
    cv2.putText(frame, f'Small: {len(small_areas)*layer}', S_text_position, font, font_scale, (0, 255, 0), 2)
    cv2.putText(frame, f'Medium: {len(medium_areas)*layer}', M_text_position, font, font_scale, (0, 255, 0), 2)
    cv2.putText(frame, f'large: {len(large_areas)*layer}', L_text_position, font, font_scale, (0, 255, 0), 2)
    plt.imshow(frame)

    cv2.imshow('Frame', frame)

    #press 'q' to exit video
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
