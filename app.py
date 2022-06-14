import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
import pickle
import pandas as pd
"""from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier"""
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

st.title('APLIKASI PENERJEMAH SANDI SEMAPHORE')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('APLIKASI PENERJEMAH SANDI SEMAPHORE')
#st.sidebar.subheader('Parameters')

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

app_mode = st.sidebar.selectbox('Pilih Mode Aplikasi',
['Info','Run on Image','Run on Video']
)

if app_mode =='Info':
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

elif app_mode =='Run on Video':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    #record = st.sidebar.checkbox("Record Video")
    #if record:
        #st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
        )

    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)

    st.sidebar.markdown('---')

    st.markdown(' ## Output')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    tfflie = tempfile.NamedTemporaryFile(delete=False)


    if not video_file_buffer:
        if use_webcam:
            #df = pd.read_csv('ds_koordinat.csv')

            #X = df.drop('class', axis=1)  # features
            #y = df['class']  # target value
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

            #pipelines = {
            #    'lr': make_pipeline(StandardScaler(), LogisticRegression()),
             #   'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
              #  'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
               # 'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
            #}

            #fit_models = {}
            #for algo, pipeline in pipelines.items():
             #   model = pipeline.fit(X_train, y_train)
            #    fit_models[algo] = model

            with open('semaphore.pkl', 'rb') as f:
                model = pickle.load(f)

            vid = cv2.VideoCapture(0)

            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_input = int(vid.get(cv2.CAP_PROP_FPS))

            st.sidebar.text('Input Video')
            st.sidebar.video(tfflie.name)
            fps = 0
            i = 0
            drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            stop = st.button('Stop Webcam')

            with kpi1:
                st.markdown("**FrameRate**")
                kpi1_text = st.markdown("0")

            with kpi2:
                st.markdown("**Semaphore Terdeteksi**")
                kpi2_text = st.markdown("Belum Terdeteksi")

            with kpi3:
                st.markdown("**Probability**")
                kpi3_text = st.markdown("0")

            with kpi4:
                st.markdown("**Image Width**")
                kpi4_text = st.markdown("0")

            st.markdown("<hr/>", unsafe_allow_html=True)


            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                prevTime = 0

                while vid.isOpened():
                    i += 1
                    ret, frame = vid.read()
                    if not ret:
                        continue

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame.flags.writeable = False

                    results = holistic.process(frame)

                    frame.flags.writeable = True
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # 1. Tangan Kanan
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                              )

                    # 2. Tangan Kiri
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                              )

                    # 3. Deteksi Pose
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                              )

                    # Export coordinates
                    try:
                        # Extract Pose landmarks
                        pose = results.pose_landmarks.landmark
                        pose_row = list(np.array(
                            [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                        # Concate rows
                        row = pose_row
                        # Make Detections
                        X = pd.DataFrame([row])
                        body_language_class = model.predict(X)[0]
                        body_language_prob = model.predict_proba(X)[0]
                        print(body_language_class, body_language_prob)

                        # Grab ear coords
                        coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                            , [640, 480]).astype(int))

                        cv2.rectangle(frame,
                                      (coords[0], coords[1] + 5),
                                      (coords[0] + len(body_language_class) * 20, coords[1] - 30),
                                      (245, 117, 16), -1)
                        cv2.putText(frame, body_language_class, coords,
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        # Get status box
                        cv2.rectangle(frame, (0, 0), (250, 60), (245, 117, 16), -1)


                        kpi2_text.write(
                            f"<h1 style='text-align: center; color: red;'>{body_language_class.split(' ')[0]}</h1>",
                            unsafe_allow_html=True)
                        kpi3_text.write(
                            f"<h1 style='text-align: center; color: red;'>{body_language_prob[np.argmax(body_language_prob)]}</h1>",
                            unsafe_allow_html=True)

                    except:
                        pass

                    currTime = time.time()
                    fps = 1 / (currTime - prevTime)
                    prevTime = currTime

                    # Dashboard
                    kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>",unsafe_allow_html=True)

                    kpi4_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

                    frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
                    frame = image_resize(image=frame, width=720)
                    stframe.image(frame, channels='BGR', use_column_width=True)

            vid.release()
        else:
            st.markdown('Masukkan Video')


    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(vid.get(cv2.CAP_PROP_FPS))

        #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        #codec = cv2.VideoWriter_fourcc('v','p','0','9')
        #codec = cv2.VideoWriter_fourcc('M','P','G','4') #MP4V-ES
        #out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

        st.sidebar.text('Input Video')
        st.sidebar.video(tfflie.name)
        fps = 0
        i = 0
        drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown("**FrameRate**")
            kpi1_text = st.markdown("0")

        with kpi2:
            st.markdown("**Detected Faces**")
            kpi2_text = st.markdown("0")

        with kpi3:
            st.markdown("**Image Width**")
            kpi3_text = st.markdown("0")

        st.markdown("<hr/>", unsafe_allow_html=True)

        with mp_pose.Pose(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence) as pose:
            prevTime = 0

            while vid.isOpened():
                i +=1
                ret, frame = vid.read()
                if not ret:
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
                results = pose.process(frame)

                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime
                #if record:
                    #st.checkbox("Recording", value=True)
                    #out.write(frame)
                #Dashboard
                kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
                #kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
                kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

                frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
                frame = image_resize(image = frame, width = 720)
                stframe.image(frame,channels = 'BGR',use_column_width=True)

            #st.text('Video Processed')

            #output_video = open('output.mp4','rb')
            #out_bytes = output_video.read()
            #st.video(out_bytes)


            vid.release()
            #out. release()

elif app_mode =='Run on Image':


    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    st.sidebar.markdown('---')

    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
    st.markdown("Masukkan Gambar . .")
    st.markdown('---')

    detection_confidence = st.sidebar.slider('Detection Confidence Value', min_value =0.0,max_value = 1.0,value = 0.5)
    st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
        st.sidebar.text('Original Image')
        st.sidebar.image(image)


        # Dashboard
        with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                min_detection_confidence=detection_confidence) as pose:

            results = pose.process(image)
            out_image = image.copy()
            #annotated_image = image.copy()
            mp_drawing.draw_landmarks(out_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imwrite(r'output.png', out_image)

            st.subheader('Output Image')
            st.image(out_image, use_column_width=True)

    else:
        st.sidebar.text('Original Image')
        #st.sidebar.image(image)

        #demo_image = DEMO_IMAGE
        #image = np.array(Image.open(demo_image))



# Watch Tutorial at www.augmentedstartups.info/YouTube