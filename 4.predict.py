import torch
import socket
import struct
import sys
import numpy as np
import time


def pointtToxyz(pointdata):

    offset = [0.06412512093828572, -0.077781122972, 0.3795246882285715]

    new_offset = [pointdata[5][0]-offset[0],
                  pointdata[5][1]-offset[1],
                  pointdata[5][2]-offset[2]]
    print(pointdata[0]-new_offset)
    row = []
    for point in pointdata:
        row.append(point[0]-new_offset[0])  # x position of point
        row.append(point[1]-new_offset[1])  # y position of point
        row.append(point[2]-new_offset[2])  # z position of point

    return row



def predict(row):

    gestures = ['backward','forward','stop','turnleft','turnright']

    model = torch.load('gestures1_model_e200.pth')
    model.to('cpu')
    model.eval()

    row = torch.Tensor([row])
    row = row.to('cpu')
    y_hat = model(row).cpu()
    y_hat = y_hat.detach().numpy()

    print("Predicted:class=%s" % (gestures[np.argmax(y_hat)]))


def tcp_server():
    serverHost = '192.168.31.39'  # localhost
    serverPort = 9090
    # Create a socket
    sSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind server to port
    try:
        sSock.bind((serverHost, serverPort))
        print('Server bind to port ' + str(serverPort))
    except socket.error as msg:
        print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
        return

    sSock.listen(10)
    print('Start listening...')
    sSock.settimeout(3.0)
    while True:
        try:
            conn, addr = sSock.accept()  # Blocking, wait for incoming connection
            break
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception:
            continue

    print('Connected with ' + addr[0] + ':' + str(addr[1]))

    while True:

        # Receiving from client
        try:
            data = conn.recv(512 * 512 * 4 + 100)
            if len(data) == 0:
                continue
            header = data[0:1].decode('utf-8')
            print('--------------------------\nCode: ' + header)

            if header == 'f':

                data_length = struct.unpack(">i", data[1:5])[0]
                N = data_length
                #print("length",N)
                point = np.frombuffer(data[5:5 + N*4], np.float32).reshape(-1,3)

                row = pointtToxyz(point)

                #print(row)

                predict(row)

        except:
            print("ignore some error")
            break

    print('Closing socket...')
    sSock.close()


if __name__ == "__main__":

    tcp_server()


