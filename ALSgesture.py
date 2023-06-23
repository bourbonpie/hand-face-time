import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)
def drawTBox(image, thumb_box_x, thumb_box_y, thumb_box_height, thumb_box_width, color, label):
    cv2.rectangle(image, (thumb_box_x, thumb_box_y), (thumb_box_x + thumb_box_width, thumb_box_y + thumb_box_height), color, 3)
    cv2.putText(image, label, (thumb_box_x, thumb_box_y - 10), cv2.FONT_HERSHEY_TRIPLEX, 1, color, 2)    
def drawIBox(image, index_box_x, index_box_y, index_box_height, index_box_width, color, label):
    cv2.rectangle(image, (index_box_x, index_box_y), (index_box_x + index_box_width, index_box_y + index_box_height), color, 3)
    cv2.putText(image, label, (index_box_x, index_box_y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2)


mpFace = mp.solutions.face_mesh
face = mpFace.FaceMesh()
mpHands = mp.solutions.hands
hands = mpHands.Hands()


mpDraw = mp.solutions.drawing_utils
mpDrawingStyles = mp.solutions.drawing_styles


with mpHands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands, mpFace.FaceMesh(
    static_image_mode=False,
    max_num_faces = 1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as face:
       
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print ("Ignoring empty camera frame.")
            continue
        
        fresults = face.process(image)
       
        if fresults.multi_face_landmarks:
            for faceLms in fresults.multi_face_landmarks:
                mpDraw.draw_landmarks(
                    image,
                    faceLms,
                    mpFace.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec = mpDrawingStyles.get_default_face_mesh_contours_style()
                )
        # Marking as not writeable to pass by reference
        hresults = hands.process(image)
       
        # Drawing hand annotations on image
        if hresults.multi_hand_landmarks:
            for handLms in hresults.multi_hand_landmarks:
                thumb_landmark = handLms.landmark[mpHands.HandLandmark.THUMB_TIP]
                index_landmark = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
               
                thumb_x, thumb_y = int(thumb_landmark.x * image.shape[1]), int(thumb_landmark.y * image.shape[0])
                index_x, index_y = int(index_landmark.x * image.shape[1]), int(index_landmark.x * image.shape[0])
               
                distance = ((thumb_x - index_x) **2 + (thumb_y - index_y) **2) **0.5
                threshold = 150
                if distance > threshold:
                    drawIBox(image, index_x, thumb_y, 100, 100, (160, 32, 255), 'L')
                else:
                    pass
                ##Somehow thumbsUp and thumbsDown are mixed up
                ##THUMB_TIP should be less than INDEX_FINGER_TIP for thumbsDown, not thumbsUp
            
                thumbsUp = (
                    handLms.landmark[mpHands.HandLandmark.THUMB_TIP].y < handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y
                    and handLms.landmark[mpHands.HandLandmark.THUMB_TIP].y < handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].y
                    and handLms.landmark[mpHands.HandLandmark.THUMB_TIP].y < handLms.landmark[mpHands.HandLandmark.RING_FINGER_TIP].y
                    and handLms.landmark[mpHands.HandLandmark.THUMB_TIP].y < handLms.landmark[mpHands.HandLandmark.PINKY_TIP].y
                )
                if thumbsUp:                               
                    cv2.circle(image, (thumb_x, thumb_y), 10, (0, 255, 0), -1)
                    drawTBox(image, thumb_x, thumb_y, 100, 100, (0, 255, 0), 'Thumbs up')
                elif not thumbsUp:
                    pass
                thumbsDown = (
                    handLms.landmark[mpHands.HandLandmark.THUMB_TIP].y > handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y
                    and handLms.landmark[mpHands.HandLandmark.THUMB_TIP].y > handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].y
                    and handLms.landmark[mpHands.HandLandmark.THUMB_TIP].y > handLms.landmark[mpHands.HandLandmark.RING_FINGER_TIP].y
                    and handLms.landmark[mpHands.HandLandmark.THUMB_TIP].y > handLms.landmark[mpHands.HandLandmark.PINKY_TIP].y
                )
                if thumbsDown:
                    thumb_landmark = handLms.landmark[mpHands.HandLandmark.THUMB_TIP]
                    cv2.circle(image, (thumb_x, thumb_y), 10, (255, 0, 0), -1)
                    drawTBox(image, thumb_x, thumb_y, 100, 100, (0, 0, 255), 'Thumbs down')                
                elif not thumbsUp or thumbsDown:
                    pass
                mpDraw.draw_landmarks(
                    image,
                    handLms,
                    mpHands.HAND_CONNECTIONS,
                    mpDrawingStyles.get_default_hand_landmarks_style(),
                    mpDrawingStyles.get_default_hand_connections_style()
                )
                
                
                print (hresults.multi_hand_landmarks)
        cv2.imshow('Hands', cv2.flip(image, 1))         
        if cv2.waitKey(1) == 27:
            break

        
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
frame = mp.Image(
    width = image.shape[1],
    height = image.shape[0],
    data= rgb_image.tobytes()
)
result = hands.process(frame)

cap.release()
cv2.destroyAllWindows()        




