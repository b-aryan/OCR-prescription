The integration of CRNN (Convolutional Recurrent Neural Network) with CTC (Connectionist Temporal Classification) forms a robust framework for text detection. By leveraging end-to-end training, the model autonomously learns to map image inputs to character sequences, eliminating the need for handcrafted feature extraction. Bidirectional LSTM layers play a pivotal role in modeling the contextual relationships within character sequences, which is essential for accurate detection. Concurrently, the convolutional layers excel at capturing spatial patterns in images, while the recurrent layers address temporal dependencies, allowing the model to balance fine-grained details with broader contextual understanding. The inclusion of CTC loss further streamlines the handling of variable-length text sequences, ensuring adaptability to diverse text structures. This synergy not only enables seamless processing of texts with varying lengths but also demonstrates strong performance in real-world applications, including complex document formats.

Given these advantages, I have designed a CRNN+CTC-based architecture tailored for text detection in PDF documents. Below is the schematic representation of the proposed network.

![310982386-514976bb-ced5-4db7-9f24-c341fb9c0969](https://github.com/user-attachments/assets/45001b02-43a4-42d0-804b-82b6940d98c3)

In essence, the CRNN architecture comprises a convolutional neural network (CNN) for spatial feature extraction and a bidirectional LSTM (BiLSTM) for contextual sequence modeling. The model generates a character sequence as output, where each predicted character corresponds to a distinct temporal step in the sequence.

Building on this framework for text detection, I conducted experiments with 11 model variations by modifying training data composition, text sequence lengths, hyperparameters, and other critical variables. After rigorous evaluation, the optimal configuration was selected as follows:


![310982922-d232a3e7-1383-4db3-bd12-d53806ee965b](https://github.com/user-attachments/assets/e735e797-6aa8-402e-bd3c-e8356312cb9f)

And here is the summary of the CRNN network:


![310983037-9d5c0e78-10f0-44ce-8988-dbf3530a8e7a](https://github.com/user-attachments/assets/66cbee84-ecb6-48d3-8d1a-ecc02be43b32)
