import socket
import network
import pickle
import cv2
import sys
sys.path.append('/home/pi/sphero-sdk-raspberrypi-python')
from sphero_sdk import SpheroRvrObserver

CAM_RESOLUTION_WIDTH = 800
CAM_RESOLUTION_HEIGHT = 600

class Movement:
    def __init__(self):
        rvr = SpheroRvrObserver()
        rvr.wake()

class Client:
	def __init__(self):
		self.client_socket = None
		self.camera = None
		self.movement = None
	
	def start(self, host_ip):
		self.movement = Movement()
		
		# socket
		self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.client_socket.connect((host_ip, network.PORT))
		
    	# camera
		self.camera = cv2.VideoCapture(0)
		self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_RESOLUTION_WIDTH)
		self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RESOLUTION_HEIGHT)

		if not self.camera.isOpened():
			print("Cannot open camera")
			exit()
	
	def step(self):
		# frame read
		did_read_frame, frame = self.cam.read()

		if not did_read_frame:
			print("Frame did not read, stream might have ended. Exiting.")
			exit(0)
		
		# networking
		network.send_object(self.client_socket, frame)
		mi = network.receive_object(self.client_socket)
		
		self.move(mi)
	
	def move(self, move_instructions):
		self.movement.rvr.drive_control.drive_forward_seconds(speed=move_instructions.speed, heading=move_instructions.heading, time_to_drive=move_instructions.time)

# main

c = Client()
c.start()

while True: c.step()
