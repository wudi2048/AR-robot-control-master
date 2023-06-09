import torch
import numpy as np

def predict(row):

    gestures = ['backward','forward','stop','turnleft','turnright']

    model = torch.load('./save/gestures1_model_e200_2022.12.07.pth')
    model.to('cpu')
    model.eval()

    row = torch.Tensor([row])
    row = row.to('cpu')
    y_hat = model(row).cpu()
    y_hat = y_hat.detach().numpy()
    # print("Predicted:class=%s" % (gestures[np.argmax(y_hat)]))
    return gestures[np.argmax(y_hat)]

if __name__ == '__main__':

    acc_num = 0
    total_num = 0
    fr = open("./rawdata/gestures3.txt","r")
    lines = fr.readlines()
    for line in lines:
        line = line.strip('\n')
        line = line.split(',')

        offset = [0.06412512093828572, -0.077781122972, 0.3795246882285715]

        new_offset = [float(line[5*3+0]) - offset[0],
                      float(line[5*3+1]) - offset[1],
                      float(line[5*3+2]) - offset[2]]
        row = []
        y = ''
        for i in range(6):
            row.append(float(line[0+i*3]) - new_offset[0])  # x position of point
            row.append(float(line[1+i*3]) - new_offset[1])  # y position of point
            row.append(float(line[2+i*3]) - new_offset[2])  # z position of point
        y = line[-1]

        y_hat = predict(row)

        total_num = total_num + 1
        if y_hat == y:
            acc_num = acc_num + 1

    print(acc_num/total_num)