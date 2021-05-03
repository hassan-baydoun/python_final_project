from contextlib import contextmanager
from io import StringIO
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
import streamlit as st
import pandas as pd
import time
from detect import detect
import os
import sys
import argparse
from PIL import Image  
import PIL

st.set_page_config(
    page_title="Hassan Baydoun - Final Project Python",
)

@contextmanager
def st_redirect(src, dst):
    '''
        Redirects the print of a function to the streamlit UI.
    '''
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    '''
        Sub-implementation to redirect for code redability.
    '''
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    '''
        Sub-implementation to redirect for code redability in case of errors.
    '''
    with st_redirect(sys.stderr, dst):
        yield

def _all_subdirs_of(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd): result.append(bd)
    return result

def _get_latest_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(_all_subdirs_of(os.path.join('runs', 'detect'), key=os.path.getmtime)

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str, default='data\images', help='source')  # file/folder, 0 for webcam
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
opt = parser.parse_args()

CHOICES = {0: "Image Upload", 1: "Upload Video"}

def _save_uploadedfile(uploadedfile):
    '''
        Saves uploaded videos to disk.
    '''
    with open(os.path.join("data", "videos",uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())


def _format_func(option):
    '''
        Format function for select Key/Value implementation.
    '''
    return CHOICES[option]

inferenceSource = str(st.sidebar.selectbox('Select Source to detect:', options=list(CHOICES.keys()), format_func=_format_func))

if inferenceSource == '0':
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=['png','jpeg', 'jpg'])
    if uploaded_file is not None:
        is_valid = True
        with st.spinner(text='In progress'):
            st.sidebar.image(uploaded_file)
            picture = Image.open(uploaded_file)  
            picture = picture.save(f'data\images\{uploaded_file.name}') 
            opt.source = f'data\images\{uploaded_file.name}'
    else:
        is_valid = False
else:
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4'])
    if uploaded_file is not None:
        is_valid = True
        with st.spinner(text='In progress'):
            st.sidebar.video(uploaded_file)
            _save_uploadedfile(uploaded_file) 
            opt.source = f'data\videos\{uploaded_file.name}'
    else:
        is_valid = False

st.title('Welcome to my Final Python Project!')
st.subheader('Presented to: Prof. Georges Salloum by Hassan BAYDOUN (192604)')

inferenceButton = st.empty()

if is_valid:
    if inferenceButton.button('Launch the Detection!'):
        with st_stdout("info"):
            detect(opt)
        if inferenceSource != '0':
            with st.spinner(text='Preparing Video'):
                for vid in os.listdir(_get_latest_folder()):
                    st.video(f'{_get_latest_folder()}\{vid}')
                st.balloons()
        else:
            with st.spinner(text='Preparing Images'):
                for img in os.listdir(_get_latest_folder()):
                    st.image(f'{_get_latest_folder()}\{img}')
                st.balloons()


