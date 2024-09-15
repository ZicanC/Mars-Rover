from __future__ import absolute_import, division, print_function

# For 3d Mapping
import cv2
import numpy as np
import open3d as o3d

import time

# For Server
import socket
import network
import socket
import pickle

# For A* Path Finding
import heapq


# For Depth detection


import os
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import torch
from torchvision import transforms
import networks
from layers import disp_to_depth
from evaluate_depth import STEREO_SCALE_FACTOR

# For Image Recognition
import base64
import requests

class Server:
    def __init__(self):
        self.connection = None
        self.server_socket = None
        self.socket_address = None
        self.PORT = 4762
        self.HEADER_LENGTH = 128
        self.HEADER_FORMAT = "utf-8"
        self.END = b'stop'
    
    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)  # "192.168.43.105"

        print("IP: ", host_ip)

        self.SOCKET_ADDRESS = (host_ip, network.PORT)

        self.server_socket.bind(self.SOCKET_ADDRESS)
        self.server_socket.listen(5)
        print("Listening.")

        self.connection, _ = self.server_socket.accept()
        print('accepted')

    def send_object(self, sock, obj):
        data = pickle.dumps(obj)        
        header = str(len(data)).encode(self.HEADER_FORMAT)
        header += b' ' * (self.HEADER_LENGTH - len(header))
        sock.send(header)       
        sock.sendall(data + self.END)
	
    def receive_object(self, sock):
        header = sock.recv(self.HEADER_LENGTH).decode(self.HEADER_FORMAT)     
        if header:
            data = b''      
            while True:
                data += sock.recv(int(header))
                if data[-len(self.END):] == self.END: break       
            return pickle.loads(data)

class Path:

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = self.create_grid(self.rows, self.cols)
        self.known_points = []

    def create_grid(self, rows, cols):
        return [[0 for _ in range(cols)] for _ in range(rows)]

    def add_point(self, point, value):
        x, y = point
        self.grid[x][y] = value
        self.known_points.append(point)

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star(self, grid, start, goal):
        rows, cols = len(grid), len(grid[0])
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor[0]][neighbor[1]] == 0:
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found
    
    def find_path(self, start = (0,0), points=[], goal = (10,10)) -> list: # Points in (x, y format)
        self.start = start
        self.goal = goal

        for point in points:
            print(point)
            self.add_point(self.grid, point, 1)

        # Find path
        path = self.a_star(self.grid, self.start, self.goal)
        print(path)
        return path

    def generate_paths(self):
    
        paths = []
        for i in range(self.rows):
            #print(i)
            if i % 2 == 0:
                paths.append(self.find_path(start=(0, i), goal=(self.cols-1, i),points=self.known_points ))
            else:
                paths.append(self.find_path(start=(self.cols-1, i), goal=(0, i), points=self.known_points  ))
        return paths
    
    def find_path_final(self, start: tuple = (0,0)):
        paths: list = self.generate_paths()
        return [j for i in paths for j in i]
        
class ImageRecog:
    def __init__(self):
        self.api_key = "3oPvOJ3IzhTGx4lMPDHLT3BlbkFJwflQxC7jarIEfphCN76R"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def scan(self, image, prompt: str = "Create a list in python format of all of the objects described in one word along with their rgb value.") -> str:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=self.generate_payload(image=image, prompt=prompt))
        
        print(type(response))
        print(response)
        return response.json()['choices'][0]['message']['content']

    def generate_payload(self, image: base64, prompt) -> dict:
        return {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self.encode_image(image)}" # Put Image Here
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500
        }
    
    def encode_image(self, image_path) -> base64:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def make_list(dict: dict):
        list = []
        for item, value in dict.items():
            list.append({'Desc': item, 'Color': value})
        return list

    def get_objects(self, image_path: str) -> list:
        results = eval(self.scan(image=image_path, prompt="You must make a list of all of the objects on the image you will receive. FORMAT LIST LIKE PYTHON DICTIONARY. Assign rgb value to each object. DON'T WRITE ANY OTHER WORDS IN YOUR RESPONSE! DO NOT ADD BACKGROUND TO LIST"))
        return self.make_list(results)

class Depth:

    def get_depth(image: np.array) -> np.array:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model_path = "C:\\Users\\chenz\\OneDrive\\Documents\\GitHub\\Final-Rover\\depthdetection\\models\\mono_1024x320"
        print("-> Loading model from ", model_path)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        print("   Loading pretrained encoder")
        encoder = networks.ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=device)

        # extract the height and width of image that this model was trained with
        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)
        encoder.to(device)
        encoder.eval()

        print("   Loading pretrained decoder")
        depth_decoder = networks.DepthDecoder(
            num_ch_enc=encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        depth_decoder.load_state_dict(loaded_dict)

        depth_decoder.to(device)
        depth_decoder.eval()
        original_width, original_height = input_image.size
        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)
        scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
        scaled_disp = scaled_disp.reshape(320,1024)

        return scaled_disp # Make this Depth Array




class SpaceConstruction:
    def create_point_cloud(depth_map, focal_length):
        height, width = depth_map.shape

        # Create a meshgrid of pixel coordinates
        i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')

        # Convert pixel coordinates to normalized image coordinates
        x = (i - width / 2) / focal_length
        y = (j - height / 2) / focal_length
        z = depth_map

        # Stack to create a 3D array of shape (height, width, 3)
        points = np.stack((x * z, y * z, z), axis=-1).reshape(-1, 3)


        return points

    def visualize_point_cloud(points):
        # Create an Open3D PointCloud object
        point_cloud = o3d.geometry.PointCloud()
        # Set points and colors
        point_cloud.points = o3d.utility.Vector3dVector(points)
        # point_cloud.colors = o3d.utility.Vector3dVector(colors)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([point_cloud])


    def create_space(self, depth_map):
        depth_map = depth_map.reshape(320,1024)
        depth_map = depth_map.astype(np.float32) / np.max(depth_map) 
        focal_length = 570.3 
        points = self.create_point_cloud(depth_map, focal_length)
        self.visualize_point_cloud(points)

class MoveInstructions:
	def __init_(self, speed, heading):
		self.speed = speed
		self.heading = heading
		self.time = 8.5/((200/255)*39.3700787402)

class Movement:
    def __init__(self) -> None:
        self.position = (0, 0)
        self.rotation = 0
    
    def change_pos(self, movement: MoveInstructions):
        while movement.rotation + self.rotation >= 360:
            movement.rotation -= 360
        self.rotation += movement.rotation

    def process_movement_data(self, next_coord: tuple) -> MoveInstructions | None: # PASS NEXT COORD
        diff = (next_coord[0] - self.position[0], next_coord[1] - self.position[1])
        movement_time = 255/100
        mi = MoveInstructions(100,0)

        if diff[0] != 0:
            if diff[0] == 1: mi.heading = 0 - self.rotation
            else:            mi.heading = 180 - self.rotation
        elif diff[1] != 0:
            if diff[1] == 1: mi.heading = 90 - self.rotation
            else:            mi.heading = 270 - self.rotation
        else:
            print("Non movement requested. Ideally, this should never show up.")
            return # We're not going anywhere

        self.change_pos(mi)
        return mi
    
def main():
    rows = int(input('Amount of rows: '))
    cols = int(input('Amount of cols: '))
    
    server = Server()
    server.start()
    iteration = 0
    recog = ImageRecog()
    objects_in_space = []
    path = Path(rows, cols)
    final_path = path.find_path_final()

    movement = Movement()

    while True:

        image = server.receive_object()
        print('Received Image')
        depth_map = Depth.get_depth(image)
        print('Made Depth')
        SpaceConstruction.create_space(depth_map)
        print('Made 3d Space')
        closest = depth_map.max()

        if closest >= 9.5:
            cv2.imwrite('current.png', image)
            objects = recog.get_objects('current.png')
            objects_in_space.append(i for i in objects)
            path.add_point(final_path[iteration+1], 2)
            final_path = path.find_path_final()

        server.send_object(movement.process_movement_data(final_path[iteration+1]))
        print("Moved")
        iteration += 1




if __name__ == '__main__':
    main()
