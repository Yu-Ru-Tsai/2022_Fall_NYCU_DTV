import cv2
import subprocess
import queue
import torch
from threading import Thread
import time

class Stream_Procss(object):
    def __init__(
        self, 
        input_url='rtmp://localhost:9099/live/test', 
        outout_url='rtmp://localhost:9097/live/test',
    ) :
        self.input_url = input_url # 'rtmp://104.210.209.70:9099/live/test'
        self.output_url = outout_url # 'rtmp://104.210.209.70:9097/live/test'

        # build the queue for processing and uploading
        self.processing_que = queue.Queue()
        self.uploading_que = queue.Queue()
        
        # get model 
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def Get_Inference_Result(self, images) :
        results = self.model(images, size=640)
        rgb_img_list = results.render()
        return rgb_img_list

    def open_ffmpeg_stream_process(self):
        args = (
            "ffmpeg -re -stream_loop -1 -f rawvideo -pix_fmt "
            "bgr24 -s 1920x1080 -i pipe:0 -pix_fmt yuv420p "
            "-f flv %s"%self.output_url
        ).split()
        return subprocess.Popen(args, stdin=subprocess.PIPE)

    def Model_Inference(self):
        while True:
            # if pre_task not finish, and no frame can be predict
            frame_list = []
            while (len(frame_list) < 32) :
                if (self.processing_que.empty()) : 
                    continue
                frame_list.append(self.processing_que.get())

            # get inference images
            result_img_list = self.Get_Inference_Result(frame_list)

            # clone the queue
            for result_img in result_img_list :
                self.uploading_que.put(result_img)

    def Upload_Frame(self) :
        # get subprocess
        self.ffmpeg_process = self.open_ffmpeg_stream_process()

        while True :
            # print('Upload_Frame:')
            # print('self.processing_que.qsize():', self.processing_que.qsize())
            # print('self.uploading_que.qsize():', self.uploading_que.qsize())
            # if pre_task not finish, and no frame can be upload
            if (self.uploading_que.empty()): 
                # time.sleep(1)
                continue 
            upload_img = self.uploading_que.get()
            self.ffmpeg_process.stdin.write(upload_img.tobytes())

        self.subprocess.stdin.close()
        self.subprocess.wait()

    def Work(self):
        Thread(target = self.Model_Inference, daemon=True).start()
        Thread(target = self.Upload_Frame, daemon=True).start()

        self.cap = cv2.VideoCapture(self.input_url)
        while True:
            ret, frame = self.cap.read()
            # Check if the frame was successfully read
            if ret:
                self.processing_que.put(frame)
        self.cap.release()


        
if __name__ == '__main__' :
    sp = Stream_Procss()
    sp.Work()