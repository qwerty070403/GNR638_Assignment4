# Assignment 4

## CNN Architecture Used (The Best Architectecture in Assignment 3)

<img src="https://github.com/user-attachments/assets/8f72a97d-c298-49d1-a3aa-87ff630e36e6" alt="image" width="900" height="200" />







### Conv Blocks:
- **Conv Block 1**: Conv (kernel size ( 1 x 1 )) → BatchNorm → Activation (LeakyReLU(0.01)) → MaxPool
- **Conv Block 2**: Conv (kernel size ( 3 x 3 )) → BatchNorm → Activation (LeakyReLU(0.01)) → MaxPool
- **Conv Block 3**: Conv (kernel size ( 3 x 3 )) → BatchNorm → Activation (LeakyReLU(0.01)) → MaxPool
- **GlobalAveragePooling2D + Fully Connected Layers (Dropout(0.3) included)**
- **Softmax Activation for Classification**

<img width="600" alt="image" src="https://github.com/user-attachments/assets/b508d3e5-23a0-430c-ab65-f3ea1dc8bcb0" />


## Experiment Details

### Objective  
To evaluate the impact of different optimizers on model performance using the best architecture from Assignment 3 for the **UCMerced Dataset** (**70% Training, 10% Validation, 20% Test**). The goal is to determine the optimizer that achieves the highest test accuracy. Additionally, we aim to analyze and visualize the loss plots to observe how quickly or slowly the models converge.  

### Experimental Setup  
- **Dataset:** UCMerced Dataset (70% Training, 10% Validation, 20% Test)  
- **Loss Function:** Categorical Cross-Entropy  
- **Evaluation Metrics:** Validation Accuracy, Test Accuracy  
- **Batch Size:** _(Specify batch size used)_  
- **Number of Epochs:** _(Specify number of epochs used)_  
- **Learning Rate:** _(Specify learning rate used or if adjusted per optimizer)_  

### Optimizers to be Compared  
The following optimizers will be tested:  

1. **Adam**  
   - Adaptive Moment Estimation (Adam) uses momentum and adaptive learning rates for efficient optimization.   
   - Default Used
  
2. **Adagrad**  
   - Adaptive Gradient Algorithm (Adagrad) assigns smaller learning rates to frequently updated parameters.  
   - Learning Rate: `[0.1,0.01,0.001]`.
       

3. **Adadelta**  
   - Adadelta dynamically adapts the learning rate based on recent updates to the model parameters.  
   - Learning Rate: `[1.0,0.1]`.


4. **RMSprop**  
   - Root Mean Square Propagation (RMSprop) adapts learning rates based on recent gradient magnitudes.  
   - Learning Rate: `[0.1,0.01,0.001]`.

  
5. **SGD with Momentum**  
   - Stochastic Gradient Descent (SGD) with momentum helps accelerate convergence by accumulating past gradients.  
   - Learning Rate: `[0.1,0.01,0.001]`, Momentum: `[0.3,0.6,0.9]`.  

---

## Results
Training was conducted for **100 epochs**.

---

## **Part 1: Adam Optimizer (Default)**

### Accuracy and Loss Plots Given Below:
![image](https://github.com/user-attachments/assets/6a9a7ecf-c47e-4ab3-907f-d82c9d5177dc)



-Best Val Accuracy: 82.81%
-Best Test Accuracy: 74.76%
-Final loss after Testing: 0.8747

---

## **Colab Links**
- **Part 1: CNN Architecture with ReLU Activation** → [Colab Notebook](https://colab.research.google.com/drive/13KHPbInQRBPFz_22U4aOXWz_WgNgXf_B#scrollTo=oWZiyaG-auD7)
- **Part 2: CNN Architecture with LeakyReLU (0.01) Activation** → [Colab Notebook](https://colab.research.google.com/drive/1fI3PFIwaAqyts_JlzwSLjfjLuoGZUoVQ?usp=sharing)
