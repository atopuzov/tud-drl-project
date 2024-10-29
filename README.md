# Tetris Deep Reinforcement Learning


# Visualization

```python
python gif.py
```

## Random agent
![](images/random.gif)

## Trained agent
![](images/tetris.gif)

# Tensorboard

```
tensorboard --logdir logs
```

## Tensorboard vscode itegration

Recommendation
```
pip install torch-tb-profiler
```

# Music

https://www.youtube.com/watch?v=NmCCQxVBfyM

https://www.youtube.com/watch?v=E8FQBjVlERk

https://www.youtube.com/watch?v=Y1TUS-yz5Yw

1min in
https://www.youtube.com/watch?v=AMcjCScsiNM

https://www.youtube.com/watch?v=s-Dq5FJEH10


https://www.youtube.com/watch?v=GfuxwAAEi8g


ffmpeg -i images/tetris.gif -vf "setpts=2*PTS" tetris.mp4


ffmpeg -i images/tetris.gif -vf "setpts=3*PTS" tetris_slow.mp4

ffmpeg -ss 00:01:00 -i audio.mp3 -t 00:00:30 -c:a aac extracted_audio.aac

ffmpeg -i tetris_slow.mp4 -i extracted_audio.aac -c:v copy -c:a aac -shortest tetris_with_audio.mp4
