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


**Best Test Accuracy**: **74.76%**

**Final loss after Testing**: **0.8747**

## **Part 2: Adagrad Optimizer**

### **2a) (lr=0.01)**

#### Accuracy and Loss Plots Given Below:
![image](https://github.com/user-attachments/assets/cb7fb23d-962f-4a5f-8e74-929c7f376464)


**Best Test Accuracy**: **74.28%**

**Final loss after Testing**: **0.8852**


### **2b) (lr=0.1)**

#### Accuracy and Loss Plots Given Below:

![image](https://github.com/user-attachments/assets/1c1c4fb0-b22f-4677-8c29-ecdb9a2f9f97)


**Best Test Accuracy**: **80.05%**

**Final loss after Testing**: **0.7207**


### **2c) (lr=0.001)**

#### Accuracy and Loss Plots Given Below:

![image](https://github.com/user-attachments/assets/7c0af3ac-c3e7-4508-a720-6b9c505ba9c8)


**Best Test Accuracy**: **47.12%**

**Final loss after Testing**: **1.7199**


## **Part 3: Adadelta Optimizer**

### **3a) (lr=0.1)**

#### Accuracy and Loss Plots Given Below:

![image](https://github.com/user-attachments/assets/bf482772-d386-42df-a253-697a517aff43)


**Best Test Accuracy**: **68.51%**

**Final loss after Testing**: **1.0084**


### **3b) (lr=1.0)**

#### Accuracy and Loss Plots Given Below:

![image](https://github.com/user-attachments/assets/6ea8b37f-b2b1-4ec4-a937-49443441da38)


**Best Test Accuracy**: **76.68%**

**Final loss after Testing**: **0.8145**


## **Part 4: RMSprop Optimizer**

### **4a) (lr=0.01)**

#### Accuracy and Loss Plots Given Below:

![image](https://github.com/user-attachments/assets/ba35ed2d-d2e7-47b3-9199-c7d05dd783f2)


**Best Test Accuracy**: **75.96%**

**Final loss after Testing**: **1.2343**


### **4b) (lr=0.001)**

#### Accuracy and Loss Plots Given Below:

![image](https://github.com/user-attachments/assets/28043739-e915-43e0-ae86-ed29b4e00add)


**Best Test Accuracy**: **75.24%**

**Final loss after Testing**: **0.8464**


### **4c) (lr=0.1)**

#### Accuracy and Loss Plots Given Below:

![image](https://github.com/user-attachments/assets/b49fb880-3c58-495f-93c3-946a168805ad)


**Best Test Accuracy**: **37.50%**

**Final loss after Testing**: **2.2755**


## **Part 5: SGD with Momentum Optimizer**

### **5a) (lr=0.1, momentum=0.9)**

#### Accuracy and Loss Plots Given Below:

![image](https://github.com/user-attachments/assets/7c07a6b4-d243-4986-b62f-ccf7f6adcddc)

**Best Test Accuracy**: **72.12%**

**Final loss after Testing**: **1.2316**


### **5b) (lr=0.1, momentum=0.6)**

#### Accuracy and Loss Plots Given Below:

![image](https://github.com/user-attachments/assets/a779c953-6bfb-4127-9305-8bc633f86bc4)

**Best Test Accuracy**: **77.40%**

**Final loss after Testing**: **0.9120**


### **5c) (lr=0.1, momentum=0.3)**

#### Accuracy and Loss Plots Given Below:

![image](https://github.com/user-attachments/assets/bb960d2b-bf8b-427b-a380-084bae4e1bb8)


**Best Test Accuracy**: **76.80%**

**Final loss after Testing**: **0.7726**


### **5d) (lr=0.01, momentum=0.9)**

#### Accuracy and Loss Plots Given Below:

![image](https://github.com/user-attachments/assets/b29f4652-428c-430c-8831-af5392883bbe)


**Best Test Accuracy**: **74.48%**

**Final loss after Testing**: **0.9295**


### **5e) (lr=0.01, momentum=0.6)**

#### Accuracy and Loss Plots Given Below:

![image](https://github.com/user-attachments/assets/32a21d1b-ce2b-4157-9d34-538d1a80d17b)


**Best Test Accuracy**: **66.24%**

**Final loss after Testing**: **1.1665**


### **5f) (lr=0.01, momentum=0.3)**

#### Accuracy and Loss Plots Given Below:

![image](https://github.com/user-attachments/assets/9659c215-48e7-46c2-a3e1-63e9e9dbb0a3)


**Best Test Accuracy**: **64.69%**

**Final loss after Testing**: **1.1000**


### **5g) (lr=0.001, momentum=0.9)**

#### Accuracy and Loss Plots Given Below:

![image](https://github.com/user-attachments/assets/e960f8ae-7705-4339-8342-5a9eb94d3203)


**Best Test Accuracy**: **62.89%**

**Final loss after Testing**: **1.2102**


### **5h) (lr=0.001, momentum=0.6)**

#### Accuracy and Loss Plots Given Below:

![image](https://github.com/user-attachments/assets/fdd74a61-eb50-486a-8db7-72fcfb85e324)


**Best Test Accuracy**: **52.58%**

**Final loss after Testing**: **1.5236**


### **5i) (lr=0.001, momentum=0.3)**

#### Accuracy and Loss Plots Given Below:

![image](https://github.com/user-attachments/assets/0c4a145f-87d3-441b-b8cc-fad8fab1f1a1)


**Best Test Accuracy**: **47.16%**

**Final loss after Testing**: **0.8025**


---

## **Colab Links**
- **Part 1: CNN Architecture with Optimiziers (Adam, Adagrad, Adadelta, RMSprop)** → [Colab Notebook](https://colab.research.google.com/drive/1EyRd79OUE4MuB0J1xVmC2kkoE3lJ8R4D?usp=sharing#scrollTo=7Rv6AANdwbrh)
- **Part 2: CNN Architecture with Optimiziers (SGD with Momentum)** → [Colab Notebook](https://colab.research.google.com/drive/14SkbfCR51F57tOec_yXtfooi8YUUOt1d#scrollTo=x0bJfgIb0aaF)
