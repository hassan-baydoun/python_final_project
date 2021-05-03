# Python (063PYTHL6) - Final Project

## Introduction

> In this presentation, i will be demonstrating a Computer Vision demo using YOLOv5 on the Microsoft COCO Dataset including close to 90 detectable objects.\
> The user can choose between detection on an image or a video.

## Installation

> Though the project will be deployed for the demo of my project, if you wish to replicate the code the following are the requirements list:
- [Python >= 3.8](https://www.python.org/downloads/)
- [GitCLI](https://cli.github.com/ "`GitCLI`")
- [Anaconda](https://www.anaconda.com/) (optional)
- streamlit -> `pip3 install streamlit`
- `git clone https://github.com/ultralytics/yolov5`
- `cd yolov5`
- `pip install -r requirements.txt`
- `rm README.md`
- `rm .gitattributes`
- `mv ./* ..`
- `rm -rfv yolov5`

or

- `git clone https://github.com/hassan-baydoun/python_final_project.git`
- `pip install -r requirements.txt`

## Run

> Run with command `streamlit run main.py`
> Image and video examples can be found in `data/images` and `data/videos`

## Code Samples
Private functions:
```python
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
    return max(all_subdirs_of('runs\\detect'), key=os.path.getmtime)

def _save_uploadedfile(uploadedfile):
    '''
        Saves uploaded videos to disk.
    '''
    with open(os.path.join("data\\videos",uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())


def _format_func(option):
    '''
        Format function for select Key/Value implementation.
    '''
    return CHOICES[option]
```

Streamlit and detection call:
```python
inferenceSource = str(st.sidebar.selectbox('Select Source to detect:', options=list(CHOICES.keys()), format_func=_format_func))

if inferenceSource == '0':
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=['png','jpeg', 'jpg'])
    if uploaded_file is not None:
        is_valid = True
        with st.spinner(text='In progress'):
            st.sidebar.image(uploaded_file)
            picture = Image.open(uploaded_file)  
            picture = picture.save(f'data/images/{uploaded_file.name}') 
            opt.source = f'data/images/{uploaded_file.name}'
    else:
        is_valid = False
else:
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4'])
    if uploaded_file is not None:
        is_valid = True
        with st.spinner(text='In progress'):
            st.sidebar.video(uploaded_file)
            _save_uploadedfile(uploaded_file)
            opt.source = f'data/videos/{uploaded_file.name}'
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
            st.warning('Video playback not available on deployed version due to resource restrictions. ')
            with st.spinner(text='Preparing Video'):
                for vid in os.listdir(_get_latest_folder()):
                    st.video(f'{_get_latest_folder()}/{vid}')
                st.balloons()
        else:
            with st.spinner(text='Preparing Images'):
                for img in os.listdir(_get_latest_folder()):
                    st.image(f'{_get_latest_folder()}/{img}')
                st.balloons()

```

***Hassan Baydoun - 2021 &infin;***


