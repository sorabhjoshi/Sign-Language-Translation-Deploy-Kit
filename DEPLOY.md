# Deploy the real-time sign translator for free

Everything below costs nothing: Kaggle (training), your own laptop (desktop app) and Hugging Face Spaces (public web link).

## 1. Train and export the model (Kaggle, free GPU) - already done for 50 signs

`slt_deploy_50signs.pt` is included in `realtime/` and `space/`. Only repeat this step to change the vocabulary or retrain.

1. Open `signlt_export_deploy.ipynb` in Kaggle (Create, New Notebook, File, Import Notebook).
2. Add Input: *Google - Isolated Sign Language Recognition* (join the competition first). Accelerator: GPU. Internet can stay off.
3. In the second cell, set `DEPLOY_SIGNS` (50 or 10), then Run All.
4. When it finishes, open the Output tab and download `slt_deploy_50signs.pt` (about 10 MB).

## 2. Run it on your computer (free, lowest latency)

```
python -m venv venv                 # Python 3.10, 3.11 or 3.12
venv\Scripts\activate              # macOS/Linux: source venv/bin/activate
pip install -r realtime/requirements.txt
cd realtime
python slt_realtime.py --model slt_deploy_50signs.pt --selftest     # no camera: checks the whole model path
python slt_realtime.py --model slt_deploy_50signs.pt --list-signs   # the words it can recognise
python slt_realtime.py --model slt_deploy_50signs.pt                # webcam, subtitles and speech
```

Keys in the window: `q` quit, `c` clear, `t` speech on/off, `l` landmarks on/off, `r` reset, space pause.
Tips: face the camera with your upper body in view, good light, sign one word, lower your hands, sign the next word.
If words are missed, try `--conf 0.35`; if it fires on random movement, raise `--conf 0.6`; if your resting hands are detected high in the frame, lower `--rest-y 1.0`.
No camera? Test with a video file: `--video clip.mp4`.

## 3. Put it online for free (Hugging Face Spaces)

1. Create a free account at huggingface.co, then New Space: SDK **Gradio**, hardware **CPU basic (free)**, visibility Public.
2. In the Space, Files, Add file, Upload files. Upload everything inside the `space/` folder (`app.py`, `slt_landmarks.py`, `slt_realtime.py`, `requirements.txt`, `packages.txt`, `README.md`) **and** `slt_deploy_50signs.pt`.
3. The Space builds by itself (5 to 10 minutes the first time). When it says Running, open the page, allow the camera and sign.
4. Share the Space URL. The voice uses each visitor's own browser speech engine (Chrome and Edge work best); press *Test voice* once to unlock audio.

Free-tier limits: the Space sleeps after a while without visitors and wakes in about a minute; two CPU cores give a few frames per second over the network, so the desktop app is smoother for a live demo.

## What it can and cannot do

* It recognises **isolated signs from a fixed vocabulary** (50 ASL signs from the Google ISLR dataset, or 10). It does not translate free sentences or other sign languages.
* On new signers the 50-sign model was about 75 to 80% accurate in your experiments. A webcam differs from the dataset recordings, so expect lower.
* Words are shown and spoken one by one. Speaking the whole phrase after a pause: `--speak sentence`.
