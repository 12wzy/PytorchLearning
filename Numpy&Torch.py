import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


# np_data = np.arange(6).reshape((2,3))
# torch_data = torch.from_numpy(np_data)
#
# tensor2array = torch_data.numpy()
#
# print "numpy array:",np_data,"\ntorch tensor:",torch_data,"\ntensor to array:",tensor2array
#
# # abs
# data = [-1, -2, 1, 2]
# tensor = torch.FloatTensor(data)  #  tensor
# print(
#     '\nabs',
#     '\nnumpy: ', np.abs(data),          # [1 2 1 2]
#     '\ntorch: ', torch.abs(tensor)      # [1 2 1 2]
# )
#
# # sin
# print '\nsin','\nnumpy:', np.sin(data),'\ntorch: ', torch.sin(tensor)
#
# # mean
# print(
#     '\nmean',
#     '\nnumpy: ', np.mean(data),         # 0.0
#     '\ntorch: ', torch.mean(tensor)     # 0.0
# )
#
#
# # matrix multiplication
# data = [[1,2], [3,4]]
# tensor = torch.FloatTensor(data)  # convert to 32 byte float tensor
# # correct method
# print(
#     '\nmatrix multiplication (matmul)',
#     '\nnumpy: ', np.matmul(data, data),     # [[7, 10], [15, 22]]
#     '\ntorch: ', torch.mm(tensor, tensor)   # [[7, 10], [15, 22]]
# )
#
# # !!!!  error method !!!!
# data = np.array(data)
# print(
#     '\nmatrix multiplication (dot)',
#     '\nnumpy: ', data.dot(data),        # [[7, 10], [15, 22]] is ok in numpy
#     '\ntorch: ', tensor.dot(tensor)     # torch convert to [1,2,3,4].dot([1,2,3,4) = 30.0
# )

tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor, requires_grad=False)

print(tensor)

print(variable.data.numpy())

x = torch.linspace(-5,5,200)
x = Variable(x)

x_np = x.data.numpy()

y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()

plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()

