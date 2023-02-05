import cv2
import time
import numpy as np

# load the COCO class names
with open('COCO_labels.txt', 'r') as f:
    class_names = f.read().split('\n')

# load the DNN model
model = cv2.dnn.readNet(model='frozen_inference_graph_V2.pb',
                        config='ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt',
                        framework='TensorFlow')

# capture the video
cap = cv2.VideoCapture('video6.mp4')
# get the video frames' width and height for proper saving of videos
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# create the `VideoWriter()` object
out = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (frame_width, frame_height))


# detect objects in each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        count = 0;
        image = frame
        image_height, image_width, _ = image.shape
        # create blob from image
        blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), swapRB=True)
        # start time to calculate FPS
        start = time.time()
        model.setInput(blob)
        output = model.forward()
        # end time after detection
        end = time.time()
        # calculate the FPS for current frame detection
        fps = 1 / (end - start)
        # loop over each of the detections
        for detection in output[0, 0, :, :]:
            # extract the confidence of the detection
            confidence = detection[2]
            # draw bounding boxes only if the detection confidence is above...
            # ... a certain threshold, else skip
            if confidence > .4:
                # get the class id
                class_id = detection[1]
                # map the class id to the class
                class_name = class_names[int(class_id) - 1]
                if class_name != "person":
                    continue
                percent = f"{(confidence * 100):.0f}"
                # get the bounding box coordinates
                box_x = detection[3] * image_width
                box_y = detection[4] * image_height
                # get the bounding box width and height
                box_width = detection[5] * image_width
                box_height = detection[6] * image_height
                # draw a rectangle around each detected object
                cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (0, 234, 255),thickness=2)


                # put the class name text on the detected object
                (w, h), _ = cv2.getTextSize(class_name+"0", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(image, (int(box_x), int(box_y )), (int(box_x+w), int(box_y -h)), (0, 234, 255), -1)
                cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 2)

                cv2.putText(image, f"{percent}%", (int(box_x + 55), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, .45,(255, 50, 0), 2)
                count += 1
                # put the FPS text on top of the frame
                cv2.putText(image, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)



        # Get the size of the text
        (text_width, text_height), _ = cv2.getTextSize(f"Bodies detected: {count}", cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)

        # Draw a filled rectangle behind the text
        cv2.rectangle(image, (5, image.shape[0] - 5), (10 + text_width, image.shape[0] - 15 - text_height), (0, 0, 0),-1)
        cv2.putText(image, f"Bodies detected: {count}", (10, image.shape[0] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7,(255, 255, 255), 1)


        cv2.imshow('video', image)
        out.write(image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()