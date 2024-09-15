import pickle

PORT = 4762
HEADER_LENGTH = 128
HEADER_FORMAT = "utf-8"
END = b"stop"

def send_object(sock, obj):
	data = pickle.dumps(obj)

	header = str(len(data)).encode(HEADER_FORMAT)
	header += b' ' * (HEADER_LENGTH - len(header))
	sock.send(header)

	sock.sendall(data + END)
	
def receive_object(sock):
	header = sock.recv(HEADER_LENGTH).decode(HEADER_FORMAT)

	if header:
		data = b''

		while True:
			data += sock.recv(int(header))
			if data[-len(END):] == END: break

		return pickle.loads(data)

class MoveInstructions:
	def __init_(self, speed, heading, time):
		self.speed = speed
		self.heading = heading
		self.time = time
