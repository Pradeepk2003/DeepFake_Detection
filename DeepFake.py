import gradio as gr
import torch
import torch.nn.functional as f
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import warnings



class Deepfake:
    def __init__(self):
        Device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.mtcnn = MTCNN(select_largest=False,
              post_process = False,
              device=Device,
              ).to(Device).eval()

        self.model = InceptionResnetV1(
            pretrained="vggface2",
            classify=True,
            num_classes=1,
            device=Device
        )

        self.checkpoint = torch.load("/home/ghost/Downloads/resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(Device)

    def predict(self,input_image: Image.Image):
        """Predict the label of the input_image"""
        self.face = self.mtcnn(input_image)
        if self.face is None:
            raise Exception('No self.face detected')
        self.face = self.face.unsqueeze(0)
        self.face = f.interpolate(self.face, size=(256, 256), mode='bilinear', align_corners=False)

        self.prev_face = self.face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
        self.prev_face = self.prev_face.astype('uint8')

        self.face = self.face.to(self.Device)
        self.face = self.face.to(torch.float32)
        self.face = self.face / 255.0
        face_image_to_plot = self.face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

        target_layers = [self.model.block8.branch1[-1]]
        cam = GradCAM(model=self.model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(0)]

        grayscale_cam = cam(input_tensor=self.face, targets=targets, eigen_smooth=True)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
        face_with_mask = cv2.addWeighted(self.prev_face, 1, visualization, 0.5, 0)

        with torch.no_grad():
            self.output = torch.sigmoid(self.model(self.face).squeeze(0))
            prediction = "real" if self.output.item() < 0.5 else "fake"
            real_prediction = 1 - self.output.item()
            fake_prediction = self.output.item()

            confidences = {
                'real': real_prediction,
                'fake': fake_prediction
            }
        return confidences, prediction

