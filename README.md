# Neural-Network-Visualization（VGG-16 image classification based on MNIST dataset）
Neural network visualization (loss-acc graph, mean gradient graph, feature map, convolution kernel, parameters)
## 1. VGG-16 architecture
Here we use the `PlotNeuralNet` drawing tool to draw our net's structure.
https://github.com/HarisIqbal88/PlotNeuralNet
<div align=center>
<img src="https://github.com/haohaohao0/Neural-network-visualization/assets/152512651/ceaabd86-cb13-4ff8-9cdf-d97435c66062" width="600" height="300">
</div>  


## 2. Parameters visualization
Here we use the `torchsummary library` to summarize our model parameters and the output sizes of each layer.  
<div align=center>
<img src="https://github.com/haohaohao0/Neural-network-visualization/assets/152512651/bb3cdc4c-608d-43d3-97dd-857d9a26a3a9" width="500" height="300">
</div>  


## 3. Loss - acc curve graph
The VGG-16 network model was trained `ten epochs` in total. After each epoch of training, the trained model was put on the verification set to calculate the accuracy.
<div align=center>
<img src="https://github.com/haohaohao0/Neural-network-visualization/assets/152512651/8026da3d-079f-4516-a887-8da8c0ce02bb" width="500" height="300">
</div> 

## 4. Average Gradient visualization
We visualize the `average gradient` of each layer in each training epoch, excluding the activation function.(Show part)
<div align=center>
<img src="https://github.com/haohaohao0/Neural-network-visualization/assets/152512651/e0e798a1-1741-4a6e-901a-827e33917f8b" width="500" height="300">
</div> 

## 5. Convolutional layer visualization 
VGG-16 has a total of `5 convolution layers` (Layer1-Layer5), and the number of convolution kernel is 64, 128, 256, 512, 512, and we will visualize each convolution kernel.(Show part)
<div align=center>
<img src="https://github.com/haohaohao0/Neural-network-visualization/assets/152512651/bc02e900-2015-4c70-b5c9-47ff034ef392" width="500" height="300">
</div> 

## 6. Feature map visualization
VGG-16 has a total of `13 convolution layers`, and we get the feature map after each convolution layer.
<div align=center>
<img src="https://github.com/haohaohao0/Neural-network-visualization/assets/152512651/2253b177-c689-49a9-b561-4117a88bc488" width="500" height="500">
</div> 


