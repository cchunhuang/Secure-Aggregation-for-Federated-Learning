import os
import sys

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.model.SimpleCNN_MNIST import trainSimpleCNNWithMNIST
from client.model.AlexNet_CIFAR10 import trainAlexNetWithCIFAR10
from client.model.ResNet18_CelebA import trainResNet18WithCelebA

dataset_folder = "./dataset"
output_folder = "./output/model"
os.makedirs(output_folder, exist_ok=True)

# # Train a simple CNN model with MNIST dataset
# print("Training a simple CNN model with MNIST dataset...")
# output_path = os.path.join(output_folder, "mnist_cnn.pth")
# result = trainSimpleCNNWithMNIST(data_folder=dataset_folder, output_model_path=output_path, train_num=1000, test_num=1000)
# print(result["train_accuracy"])
# print(result["train_loss"])
# print(result["test_accuracy"])

# # Load a pre-trained model, fine-tune it, and save the updated model
# print("Loading a pre-trained model, fine-tuning it, and saving the updated model...")
# input_path = output_path
# output_path = os.path.join(output_folder, "mnist_cnn_updated.pth")
# result = trainSimpleCNNWithMNIST(data_folder=dataset_folder, input_model_path=input_path, output_model_path=output_path)

# # Train an AlexNet model with CIFAR-10 dataset
# print("Training an AlexNet model with CIFAR-10 dataset...")
# output_path = os.path.join(output_folder, "cifar10_alexnet.pth")
# # result = trainAlexNetWithCIFAR10(data_folder=dataset_folder, output_model_path=output_path, epoch=2)
# result = trainAlexNetWithCIFAR10(data_folder=dataset_folder, output_model_path=output_path, train_num=5000, test_num=5000)
# print(result["train_accuracy"])
# print(result["train_loss"])
# print(result["test_accuracy"])

# # Load a pre-trained model, fine-tune it, and save the updated model
# print("Loading a pre-trained model, fine-tuning it, and saving the updated model...")
# input_path = output_path
# output_path = os.path.join(output_folder, "cifar10_alexnet_updated.pth")
# result = trainAlexNetWithCIFAR10(data_folder=dataset_folder, input_model_path=input_path, output_model_path=output_path)

# Train a ResNet18 model with CelebA dataset
print("Training a ResNet18 model with CelebA dataset...")
dataset_folder = './dataset/CelebA/img_align_celeba'
output_path = os.path.join(output_folder, "celeba_resnet18.pth")
label_file = './dataset/CelebA/list_attr_celeba.txt'
result = trainResNet18WithCelebA(data_folder=dataset_folder, label_file=label_file, output_model_path=output_path, train_num=5000, test_num=1000)
print(result["train_accuracy"])
print(result["train_loss"])
print(result["test_accuracy"])

# # Load a pre-trained model, fine-tune it, and save the updated model
# print("Loading a pre-trained model, fine-tuning it, and saving the updated model...")
# input_path = output_path
# output_path = os.path.join(output_folder, "celeba_resnet18_updated.pth")
# result = trainResNet18WithCelebA(data_folder=dataset_folder, label_file=label_file, eval_file=eval_file, input_model_path=input_path, output_model_path=output_path)