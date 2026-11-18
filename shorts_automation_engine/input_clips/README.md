# input_clips

Drop **6 raw clips** here (`.mp4 .mov .m4v .mkv .webm .avi`), then run:

```bash
python main.py --local-clips                      # Claude watches the clips, picks the topic, ranks them
python main.py "craziest goals" --local-clips     # same, but you set the topic
python main.py "craziest goals" --local-clips --keep-order
```

* Up to 10 clips are considered; Claude picks the best 6 and decides which is #1.
* `--keep-order` pins the ranking yourself: name files `1_xxx.mp4 … 6_xxx.mp4`
  (number = rank), or any names - then alphabetical order is the play order (first file = #6).
* Any aspect ratio works: portrait clips are center-cropped, landscape/square clips
  get a blurred-background fill (`--fit crop|blur|smart`).
