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


**Best Val Accuracy**: **82.81%**

**Best Test Accuracy**: **74.76%**

**Final loss after Testing**: **0.8747**

## **Part 2: Adagrad Optimizer**

### **2a: (lr=0.01)**

#### Accuracy and Loss Plots Given Below:

**Best Test Accuracy**: **74.28%**

**Final loss after Testing**: **0.8852**


### **2b: (lr=0.1)**

#### Accuracy and Loss Plots Given Below:

**Best Test Accuracy**: **80.05%**

**Final loss after Testing**: **0.7207**


### **2c: (lr=0.001)**

#### Accuracy and Loss Plots Given Below:

**Best Test Accuracy**: **47.12%**

**Final loss after Testing**: **1.7199**


## **Part 3: Adadelta Optimizer**

### **3a: (lr=0.1)**

#### Accuracy and Loss Plots Given Below:

**Best Test Accuracy**: **68.51%**

**Final loss after Testing**: **1.0084**


### **3b: (lr=1.0)**

#### Accuracy and Loss Plots Given Below:

**Best Test Accuracy**: **76.68%**

**Final loss after Testing**: **0.8145**


## **Part 4: RMSprop Optimizer**

### **4a: (lr=0.01)**

#### Accuracy and Loss Plots Given Below:

**Best Test Accuracy**: **75.96%**

**Final loss after Testing**: **1.2343**


### **4b: (lr=0.001)**

#### Accuracy and Loss Plots Given Below:

**Best Test Accuracy**: **75.24%**

**Final loss after Testing**: **0.8464**


### **4c: (lr=0.1)**

#### Accuracy and Loss Plots Given Below:

**Best Test Accuracy**: **37.50%**

**Final loss after Testing**: **2.2755**


## **Part 5: SGD with Momentum Optimizer**

### **5a: (lr=0.1),(momentum=0.9)**

#### Accuracy and Loss Plots Given Below:

**Best Test Accuracy**: **72.12%**

**Final loss after Testing**: **1.2316**


### **5b: (lr=0.1),(momentum=0.6)**

#### Accuracy and Loss Plots Given Below:

**Best Test Accuracy**: **77.40%**

**Final loss after Testing**: **0.9120**


### **5c: (lr=0.1),(momentum=0.3)**

#### Accuracy and Loss Plots Given Below:

**Best Test Accuracy**: **76.80%**

**Final loss after Testing**: **0.7726**


### **5d: (lr=0.01),(momentum=0.9)**

#### Accuracy and Loss Plots Given Below:

**Best Test Accuracy**: **74.48%**

**Final loss after Testing**: **0.9295**


### **5e: (lr=0.01),(momentum=0.6)**

#### Accuracy and Loss Plots Given Below:

**Best Test Accuracy**: **66.24%**

**Final loss after Testing**: **1.1665**


### **5f: (lr=0.01),(momentum=0.3)**

#### Accuracy and Loss Plots Given Below:

**Best Test Accuracy**: **64.69%**

**Final loss after Testing**: **1.1000**


### **5g: (lr=0.001),(momentum=0.9)**

#### Accuracy and Loss Plots Given Below:

**Best Test Accuracy**: **62.89%**

**Final loss after Testing**: **1.2102**


### **5h: (lr=0.001),(momentum=0.6)**

#### Accuracy and Loss Plots Given Below:

**Best Test Accuracy**: **52.58%**

**Final loss after Testing**: **1.5236**


### **5i: (lr=0.001),(momentum=0.3)**

#### Accuracy and Loss Plots Given Below:

**Best Test Accuracy**: **47.16%**

**Final loss after Testing**: **0.8025**


---

## **Colab Links**
- **Part 1: CNN Architecture with ReLU Activation** → [Colab Notebook](https://colab.research.google.com/drive/13KHPbInQRBPFz_22U4aOXWz_WgNgXf_B#scrollTo=oWZiyaG-auD7)
- **Part 2: CNN Architecture with LeakyReLU (0.01) Activation** → [Colab Notebook](https://colab.research.google.com/drive/1fI3PFIwaAqyts_JlzwSLjfjLuoGZUoVQ?usp=sharing)
