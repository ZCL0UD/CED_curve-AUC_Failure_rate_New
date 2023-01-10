import math
from re import X
from tkinter import Y
import cv2
import dlib
import imutils
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def find_landmarks():
    landmark_points_68 = [162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389, 71, 63, 105, 66, 107, 336,
                          296, 334, 293, 301, 168, 197, 5, 4, 75, 97, 2, 326, 305, 33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373,
                          380, 61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14, 87]
    return landmark_points_68


def read_landmarks(pts_file_path):
    points = []
    rows = open(pts_file_path).read().strip().split("\n")
    rows = rows[3:-1]  # take only the 68-landmarks
    for row in rows:
        # break the row into the filename and bounding box coordinates
        row = row.strip()  # remove blanks at the beginning and at the end
        row = row.split(" ")  # one space
        row = np.array(row, dtype="float32")  # convert list into float32
        (startX, startY) = row
        points.append([startX, startY])
        # points.extend(row)
    # convert a List into array of float32
    points = np.array(points, dtype=np.float32).reshape((-1, 2))  # (68, 2)
    return points


def two_points_distance(x_ini, y_ini, x_fin, y_fin):
    first_term = x_ini - x_fin
    second_term = y_ini - y_fin
    return math.sqrt(math.pow(first_term, 2) + math.pow(second_term, 2))


def calculate_DF(width_face, height_face):
    return math.sqrt(width_face*height_face)



def my_NME_image(pts_pred, pts_true):
    nme = 0
    N = 68
    inter_ocular = np.linalg.norm(pts_true[37-1,:]-pts_true[46-1,:])
    nme = np.sum(np.linalg.norm(pts_pred - pts_true, axis=1)) / (inter_ocular*N)
    return nme

def kevin_vinueza(path_imagen):

    import facial_keypoints_detecter as fkd
    y_pred = []
    

    net = fkd.model.Net()
    net.load_model('model_kevin.pt')
    net.eval()
    keypoints, images = net.apps.detect_facial_keypoints(path_imagen, plot_enabled = False ,figsizeScale = 1)  
       
    for i in range(0, 68):
        if len(keypoints)>0:
            x, y = keypoints[0][0][i][0].item(), keypoints[0][0][i][1].item() 
            if(x<0):
                x=0.0
            if(y<0):
                y=0.0
        else:
            x, y =0.0,0.0
        y_pred.append([x, y])
    
        
    if len(keypoints)>0: 
        y_pred = np.array(y_pred, dtype=np.float32).reshape((-1, 2))
    
    return y_pred



def Dlib_Model(path_imagen):
    cap = cv2.imread(path_imagen)
    # save the predicted values for the model
    y_pred = []
    face_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    frame = imutils.resize(cap, width=1000)
    gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
    coordinates_bboxes = face_detector(gray, 1)
    for c in coordinates_bboxes:
        x_ini, y_ini, x_fin, y_fin = c.left(), c.top(), c.right(), c.bottom()
        shape = predictor(gray, c)
    for i in range(0, 68):
        x, y = shape.part(i).x, shape.part(i).y
        y_pred.append([x, y])
      
    width_face = two_points_distance(x_ini, y_ini, x_fin, y_fin)
    height_face = two_points_distance(x_ini, y_ini, x_fin, y_fin)
    Df = calculate_DF(width_face, height_face)
    y_pred = np.array(y_pred, dtype=np.float32).reshape((-1, 2))
    return y_pred , Df


def karelis(path_imagen):
    y_pred=[]
    landmark_points_68 = find_landmarks()
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    frame = cv2.imread(path_imagen)
    face_detector = dlib.get_frontal_face_detector()
    frame = imutils.resize(frame,width=720)
    gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    coordinates_bboxes= face_detector(gray,1)
    #Inicializar las variables con valores por defecto
    x_ini=0
    y_ini=0
    x_fin=0
    y_fin=0

    for c in coordinates_bboxes:
        x_ini,y_ini,x_fin,y_fin = c.left(),c.top(),c.right(),c.bottom()

    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height ,width,_ =frame_rgb.shape
            results = face_mesh.process(frame_rgb)
            if results.multi_face_landmarks is not None:
                for face_landmarks in results.multi_face_landmarks:
                    for index in landmark_points_68:
                        x = int(face_landmarks.landmark[index].x * width)
                        y = int(face_landmarks.landmark[index].y * height)
                        xvalue = face_landmarks.landmark[index].x * width
                        yvalue = face_landmarks.landmark[index].y * height
                        y_pred.append([xvalue, yvalue])

    width_face= two_points_distance(x_ini,y_ini,x_fin,y_fin)
    height_face= two_points_distance(x_ini,y_ini,x_fin,y_fin)

    Df = calculate_DF(width_face,height_face)

    return y_pred , Df       

def bryan(path_image):
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Hyperparameters """
    image_h = 512
    image_w = 512

    """ Paths """
    dataset_path = path_image
    model_path = os.path.join("files", "model.h5")

    """ Load the model """
    model = tf.keras.models.load_model(model_path)
    # model.summary()
    """ Prediction """

    """ Reading the image """
    name = dataset_path.split("/")[-1].split(".")[0]
    image = cv2.imread(dataset_path, cv2.IMREAD_COLOR)
    image_x = image
    image = cv2.resize(image, (image_w, image_h))
    imagen_a=image
    image = image/255.0  # (512, 512, 3)
    image = np.expand_dims(image, axis=0)  # (1, 512, 512, 3)
    
    image = image.astype(np.float32)

    """ Facial Size"""
    face_detector = dlib.get_frontal_face_detector()
    coordinates_bboxes = face_detector(imagen_a, 1)
    for c in coordinates_bboxes:
        x_ini, y_ini, x_fin, y_fin = c.left(), c.top(), c.right(), c.bottom()
    width_face = two_points_distance(x_ini, y_ini, x_fin, y_fin)
    height_face = two_points_distance(x_ini, y_ini, x_fin, y_fin)
    df = calculate_DF(width_face, height_face)

    """ Prediction """
    pred = model.predict(image, verbose=0)[0]
    pred = pred.astype(np.float32)
    y_pred = []
    h, w, _ = image_x.shape
    for i in range(0, 66, 4):
        x = int(pred[i] * w)
        y = int(pred[i+1] * h)
        y_pred.append([x,y])
    for i in range(66, 76, 2):
        x = int(pred[i] * w)
        y = int(pred[i+1] * h)
        y_pred.append([x,y])
    for i in range(84, 94, 2):
        x = int(pred[i] * w)
        y = int(pred[i+1] * h)
        y_pred.append([x,y])
    for i in range(102, 110, 2):
        x = int(pred[i] * w)
        y = int(pred[i+1] * h)
        y_pred.append([x,y])
    for i in range(116, 126, 2):
        x = int(pred[i] * w)
        y = int(pred[i+1] * h)
        y_pred.append([x,y])
    for i in range(132, 136, 2):
        x = int(pred[i] * w)

def charlie(path_imagen):
    # Load the pre-trained model
    landmark_detector = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel("LFBmodel.yaml")

    # Load the image
    image = cv2.imread(path_imagen)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml").detectMultiScale(gray)

    # Initialize a list to store the landmarks for each face
    y_pred = []

    resulDf=0
    # Iterate through each face detected
    for (x, y, w, h) in faces:

        # Calculate the width and height of the face
        width_face = two_points_distance(x, y, x + w, y + h)
        height_face = two_points_distance(x, y, x + w, y + h)

        # Calculate the facial form index (DF)
        resulDf= calculate_DF(width_face, height_face)
        # Detect landmarks on the grayscale image
        _, landmarks = landmark_detector.fit(gray, np.array([(x, y, w, h)]))

        # Draw a circle at each landmark point
        y_pred_face = []
        for x, y in landmarks[0][0]:        
            y_pred_face.append([x, y])
        y_pred.append(y_pred_face)

    # Convert the list y_pred to a NumPy array with shape (-1, 68, 2)
    y_pred = np.array(y_pred, dtype=np.float32).reshape((-1, 68, 2))

    Df= resulDf
    return y_pred , Df       
