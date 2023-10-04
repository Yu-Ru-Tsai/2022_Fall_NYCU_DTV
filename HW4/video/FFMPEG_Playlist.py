import subprocess
import argparse
import os

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    args = parser.parse_args()
    return args

def create_master_playlist(video):
    command = "ffmpeg -re -stream_loop 1 -i {video} -vcodec libx264 \
              -pix_fmt yuv420p -g 25 -keyint_min 25 \
              -sc_threshold 0 -hls_time 2 \
              -b:v:0 800k  -filter:v:0 scale=640:360 \
              -b:v:1 1200k -filter:v:1 scale=842:480 \
              -b:v:2 2400k -filter:v:2 scale=1280:720 \
              -b:v:3 4800k -filter:v:3 scale=1920:1080 \
              -map 0:v -map 0:v -map 0:v -map 0:v \
              -f hls \
              -var_stream_map 'v:0 v:1 v:2 v:3' \
              -master_pl_name master.m3u8 \
              -hls_segment_filename stream_%v/data%03d.ts \
              -use_localtime_mkdir 1 \
              -hls_flags independent_segments\
              -hls_playlist_type event\
              stream_%v.m3u8".format(video=video)
    subprocess.call(command,shell=True)

if __name__ == "__main__":
  args = arg_parse()
  create_master_playlist(args.input_file)
