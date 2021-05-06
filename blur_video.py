import cv2 as cv
import numpy as np
import coremltools as ct
import matplotlib.pyplot as plt

class ImageSegmentor:
    def __init__(self):
        input_file_name='mkbhd_m1_clip'
        # input_file_name='ranjeet_arm_clip'
        self.input_video_path = '/Users/akshit/Downloads/{}.mp4'.format(input_file_name)
        # self.input_video_path = '/Users/akshit/Downloads/mkbhd_m1_clip.mp4'
        self.blurred_output_video_path = '{}_blurred.avi'.format(input_file_name)
        self.plot_video_path = '{}_plot.avi'.format(input_file_name)
        
        self.model_path = "models/DeeplabV3.mlmodel"
        self.seg_model = ct.models.MLModel(self.model_path)
        self.model_input_dim = 513
        self.do_plot = False

    def compute_border_dims(self, img_dims):
        if len(img_dims)==3:
            h, w, _ = img_dims
        else:
            h, w = img_dims
        dim = max(w,h)
        bl = (dim-w)//2
        br = dim-(w+bl)
        bt = (dim-h)//2
        bb = dim-(h+bt)
        return bt, bb, bl, br

    def pad_to_square(self, img):
        bt, bb, bl, br = self.compute_border_dims(img.shape)
        padded_img = cv.copyMakeBorder(img, bt, bb, bl, br, cv.BORDER_CONSTANT)
        return padded_img

    def resize_with_same_aspect_ratio(self, img, out_max_dim):
        h, w, c = img.shape
        resize_ratio = 1.0 * out_max_dim / max(w, h)
        target_size = (int(resize_ratio * w), int(resize_ratio * h))
        return cv.resize(img, target_size)

    def resize_to_model_input_dims(self, img):
        bounded_img = self.resize_with_same_aspect_ratio(img, self.model_input_dim)
        square_img = self.pad_to_square(bounded_img)
        rgb_image = cv.cvtColor(square_img, cv.COLOR_BGR2RGB)
        return rgb_image        

    def adapt_to_model_input(self, img):
        rgb_image = self.resize_to_model_input_dims(img)
        # im_normed = padded_im.astype(np.uint8)#(padded_im/255.).astype(np.float32)
        batched_im = np.expand_dims(rgb_image, 0)
        return batched_im.astype(np.float32)

    def adapt_to_original_input(self, img, target_dims):
        h,w,c = target_dims
        # resize_ratio = max(h,w)/img.shape[0]
        target_size = max(h,w),max(h,w)
        resized_img = cv.resize(img, target_size)
        # crop
        bt, bb, bl, br = self.compute_border_dims(target_dims)
        cropped_img = resized_img[bt:h+bt, bl:w+bl]
        return cropped_img

    def predict_seg_map(self, img_tensor):
        preds=self.seg_model.predict({'ImageTensor':img_tensor})#, useCPUOnly=True)
        seg_map = preds['SemanticPredictions'].astype(np.uint8).squeeze()
        seg_map[seg_map!=15] = 0
        seg_map[seg_map==15] = 1
        return seg_map

    def segement_image(self, in_img):
        model_adapted_img = self.adapt_to_model_input(in_img)
        seg_map =  self.predict_seg_map(model_adapted_img)
        seg_map_orig_res = self.adapt_to_original_input(seg_map, in_img.shape)
        seg_smooth = cv.blur(seg_map_orig_res.astype(np.float32), (10,10))
        return seg_smooth

    # seg_map: [[np.float]]
    def blur_background(self, in_img, seg_map):
        blur_img = cv.GaussianBlur(in_img,(21,21),0)
        blur_bg_composed = (in_img*np.expand_dims(seg_map, axis=2) + blur_img*np.expand_dims(1-seg_map, axis=2)).astype(np.uint8)
        return blur_bg_composed

    def segment_video(self):
        input_video = cv.VideoCapture(self.input_video_path)
        # https://stackoverflow.com/questions/49025795/python-opencv-video-getcv2-cap-prop-fps-returns-0-0-fps
        input_fps = input_video.get(cv.CAP_PROP_FPS)
        # https://docs.opencv.org/master/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
        input_frame_size = (
            int(input_video.get(cv.CAP_PROP_FRAME_WIDTH)), 
            int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT)))
        
        print("reading video: {} with fps: {} with frame size: {}".format(self.input_video_path, input_fps, input_frame_size))
        # https://www.geeksforgeeks.org/saving-a-video-using-opencv/
        blurred_video = cv.VideoWriter(
            self.blurred_output_video_path, 
            cv.VideoWriter_fourcc(*'MJPG'), 
            input_fps,
            input_frame_size)
        if self.do_plot:
            # https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib#
            plt.ion()
            fig = plt.figure(figsize=(6, 10), dpi=80, tight_layout=True)
            plt.show()
            # can be different than what is set
            plot_video_size = tuple((2*fig.get_size_inches()*fig.get_dpi()).astype(np.int32))
            plot_video = cv.VideoWriter(
                self.plot_video_path, 
                cv.VideoWriter_fourcc(*'MJPG'),
                input_fps,
                plot_video_size)
        
        seconds_to_process = input_video.get(cv.CAP_PROP_FRAME_COUNT)
        frames_to_process = int(input_video.get(cv.CAP_PROP_FRAME_COUNT))
        # seconds_to_process = 1
        # frames_to_process = int(input_fps*seconds_to_process)

        # bring to front
        while (input_video.isOpened()):
            _, frame = input_video.read()
            seg_map = self.segement_image(frame)
            assert seg_map.shape[:2]==frame.shape[:2]
            blur_bg = self.blur_background(frame, seg_map)
            blurred_video.write(blur_bg)
            frames_to_process-=1
            print("{} frames remaining...".format(frames_to_process))
            if frames_to_process <=0:
                break

            # show original video
            #cv.imshow('Frame', frame)
            # cv.setWindowProperty('Frame', cv.WND_PROP_TOPMOST, 1)

            #show segmentation map
            #plt_im.set_data(seg_map)
            if self.do_plot:            
                plt.subplot(311),plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB)),plt.title('Original')
                plt.xticks([]), plt.yticks([])
                plt.subplot(312),plt.imshow(cv.cvtColor(blur_bg, cv.COLOR_BGR2RGB)),plt.title('Blurred bg')
                plt.xticks([]), plt.yticks([])
                plt.subplot(313),plt.imshow(seg_map),plt.title('Seg map')
                plt.xticks([]), plt.yticks([])
                
                # plt.imshow(seg_map)
                plt.draw()
                plt.pause(0.001)

                # https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
                data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                data = cv.cvtColor(data, cv.COLOR_BGR2RGB)
                plot_video.write(data)
                # plt.show(block=False)
                # cv.setWindowProperty('Segmentation', cv.WND_PROP_TOPMOST, 1)

            # define q as the exit button
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
            
        input_video.release()
        blurred_video.release()
        if self.do_plot:
            plot_video.release()
        
        cv.destroyAllWindows()

def main():
    img_segmentor = ImageSegmentor()
    img_segmentor.segment_video()

if __name__ == "__main__":
    main()