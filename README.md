# Speaker and Phrase Recognition with CNNs

This project implements a **biometric voice verification system** using **Convolutional Neural Networks (CNNs)** accelerated with GPU.  
The system can recognize both **who is speaking** and **what phrase is being spoken**, based on audio recordings preprocessed into Mel spectrograms.

The work was developed as an academic project and serves as a practical example of applying **Deep Learning** to audio processing tasks.

---

## ✨ Main Features
- **Two independent CNN models**:
  - Phrase recognition (5 predefined phrases).  
  - Speaker recognition (over 800 participants).  
- **Automatic preprocessing**:
  - Audio normalization.  
  - Conversion into log-scaled Mel spectrograms.  
- **Flexible training** with callbacks (`EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`).  
- **Usage modes**:
  - Train models.  
  - Evaluate on test set.  
  - Random recognition of phrase and/or speaker.  

---
