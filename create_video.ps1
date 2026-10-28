param(
	[string]$Input_Path
	)

py src/imager.py --input $Input_Path --output $Input_Path/frames --rotating-frame --mark-depinned --save

ffmpeg -start_number 1 -framerate 30 -i $Input_Path/frames/frame_%08d.png -c:v h264 -pix_fmt yuv420p $Input_Path/video.mp4