# 🐾 Rare Species Image Classification with Deep Learning

## 📘 Overview

This project aims to develop a **deep learning-based computer vision model** capable of accurately predicting the **species of an animal** from an input image. Leveraging **Convolutional Neural Networks (CNNs)**—which are particularly well-suited for image classification—the model is trained on a **diverse dataset of animal images** representing multiple species.

The primary goal is to enable **rapid and automated species identification**, reducing reliance on manual observation and expert intervention.

---

## ⚙️ Methodology

We followed a structured ML pipeline:

1. **Exploratory Data Analysis (EDA)** – to understand dataset characteristics and guide modeling decisions.
2. **Preprocessing** – cleaned the dataset by removing noisy or irrelevant images.
3. **Data Augmentation** – generated synthetic images to address class imbalance and expand the training set.
4. **Modeling** – began with handcrafted CNNs, later adopting **transfer learning** with pre-trained models to boost performance.

---

## 🏆 Best Model: BiGAN (256×256) with Classic Augmentation

After analyzing performance metrics across several configurations, the **BiGAN model with classic augmentation and synthetic images at 256×256 resolution** clearly emerged as the **top-performing approach**.

- **Test Accuracy**: 0.8457  
- **F1 Score**: 0.8436  
- **Lowest loss** across training, validation, and test sets

Alternative BiGAN versions using lower resolutions (224×224 and 64×64) showed progressively weaker results, emphasizing the importance of **higher image resolution** and **appropriate augmentation strategies**.

### 📊 Dataset Comparison

| Dataset           | Accuracy | Test Loss |
|------------------|----------|-----------|
| Cleaned Dataset  | 0.8264   | 0.6871    |
| Original Dataset | 0.8175   | 0.7007    |

The **Cleaned Dataset** led to improved generalization, showing that **data cleaning effectively removed noise** that could hinder classification performance.

---

## 🚧 Limitations and Future Work

To continue improving this project, we propose the following directions:

- **Image Preprocessing Enhancements**: Explore grayscale conversion to assess the impact of color on model performance.
- **Transformer Variants**: Since a transformer-based model outperformed others, testing additional transformer architectures may yield even better results.
- **Cross-validation**: Implement **k-fold cross-validation** to improve the reliability of model evaluation, especially in scenarios with limited data.

---

## ✅ Conclusion

This project demonstrates the critical role of both **architecture selection** and **data preprocessing** in deep learning applications for specialized domains like rare species identification.

Our combination of:
- Efficient deep learning architecture,
- Strategic data augmentation,
- Systematic hyperparameter tuning,

…produced a **robust image classification system** capable of identifying rare species with high accuracy.

These findings lay a solid foundation for future **biodiversity conservation efforts**, enabling automated identification of endangered and threatened species in real-world scenarios where image quality and visual distinctiveness may vary.


