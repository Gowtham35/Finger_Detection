import cv2
import mediapipe as mp
import math


class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []
        
        self.hand = self.finger = None 
        self.text = ""
        

    def findHands(self, img, draw=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {"lmList":[]}
                
                ## lmList
                xList, yList = [], []

                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    myHand["lmList"].append([px, py, pz])
                    xList.append(px)
                    yList.append(py)
 
                myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    if self.hand and self.finger is not None:
                        ftypes = {0:"Thumb", 1:"Index", 2:"Middle", 3:"Ring", 4:"Pinky"}
                        ftype = ftypes[self.finger]
                        
                        if self.hand == "Left":
                            self.finger = 4 - self.finger
                        else:
                            self.finger += 5
                            
                        self.text = f"{self.finger} - {self.hand}-{ftype}"
                        print(self.text)

                    xmin, ymin = min(xList), min(yList)
                    cv2.putText(img, self.text, (xmin - 30, ymin - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
                    
                    self.hand = self.finger = None
                    self.text = ""
                    
        if draw:
            return allHands, img
        else:
            return allHands
        

    def fingersUp(self, myHand):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:
            fingers = []
            # Thumb
            if myHandType == "Left":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        if fingers.count(0) == 1:
            self.hand = myHandType
            self.finger = fingers.index(0)

        return fingers


    def main(self):
        cap = cv2.VideoCapture(0)
    
        detector = HandDetector(detectionCon=0.8, maxHands=2)
    
        while True:
            # Get image frame
            success, img = cap.read()
            img = cv2.flip(img, 1)
            # Find the hand and its landmarks
            hands, img = detector.findHands(img)  # with draw
            # hands = detector.findHands(img, draw=False)  # without draw

            for hand in hands:
                _fingers = detector.fingersUp(hand)

            # Display
            cv2.imshow("Image", img)
        
            key = cv2.waitKey(1)
            if key == ord('q'):
                return

        

if __name__ == "__main__":

    # from filename import HandDetector
    handObject = HandDetector()
    handObject.main()

