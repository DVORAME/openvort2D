$Path = if ($env:PATH) { $env:PATH } else { 'output' }

py src/imager.py --input $Path --output_dir $Path/frames --rotating_frame --mark_depinned --save

ffmpeg -start_number 1 -framerate 30 -i $Path/frames/frame_%08d.png -c:v h264 -pix_fmt yuv420p $Path/video.mp4