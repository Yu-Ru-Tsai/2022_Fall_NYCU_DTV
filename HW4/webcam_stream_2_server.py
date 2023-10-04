import subprocess

# Start an ffmpeg process to capture the webcam video and send it to the RTMP server
# ffmpeg -re -stream_loop -1 -i project_view.mp4 -c:v copy -c:a copy -f flv rtmp://your_ip:1935/live/test
"""
    mp4_file:
    [
        "ffmpeg", "-f", "avfoundation", "-i", "./test.mp4", "-vcodec", "h264", "-acodec", "aac", 
        "-f", "flv", "rtmp://104.210.209.70:9099/live/test"
    ]
    #ffmpeg -f avfoundation -framerate 30 -i "2" -s 1920x1080 -f flv rtmp://104.210.209.70:9099/live/test

"""
process = subprocess.Popen(
    [
        "ffmpeg", "-f", "avfoundation", "-framerate", "30", "-i", "2",
        "-s", "1920x1080",
        "-f", "flv", "rtmp://104.210.209.70:9099/live/test"
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

# Wait for the process to finish
process.wait()

# Print the output of the process
print(process.stdout.read().decode())