## 📂 Files in This Repository
- **`weatherforecasting_tcn.ipynb`** → Jupyter Notebook containing the model training and evaluation.
- **`GlobalWeatherRepository.csv`** → The dataset used for training the model.

## DESCRIPTION
- This project focuses on building a **Temporal Convolutional Network (TCN)** for weather forecasting. The model predicts three key weather parameters—**temperature**, **wind speed**, and **precipitation**—for the next 7 days.The TCN architecture leverages dilated convolutions and residual connections to capture temporal dependencies in the data, making it well-suited for time-series forecasting tasks.

- ## Key Features
- **Data Preprocessing**: Handles missing values, scales features, and creates sequences for time-series forecasting.
- **TCN Model**: Implements a deep learning model with dilated convolutions and residual connections for accurate predictions.
- **Hyperparameter Tuning**: Uses Keras Tuner to optimize model hyperparameters (e.g., filters, kernel size, learning rate).
- **Evaluation Metrics**: Evaluates model performance using RMSE, MSE, MAE, and R² for each target variable.
- **Visualization**: Generates plots comparing actual vs predicted values for temperature, wind speed, and precipitation.

## Results
The best model achieves:
- **Overall R²**: 0.9151
- **Temperature R²**: 0.8877
- **Wind Speed R²**: 0.8942
- **Precipitation R²**: 0.9633

## Technologies Used
- **Python**: Primary programming language.
- **TensorFlow/Keras**: For building and training the TCN model.
- **Pandas/NumPy**: For data preprocessing and manipulation.
- **Matplotlib/Seaborn**: For data visualization.
- **Scikit-learn**: For evaluation metrics and scaling.

## Future Work
- Incorporate additional features (e.g., humidity, pressure) to improve predictions.
- Experiment with hybrid architectures (e.g., TCN-LSTM).
- Deploy the model in a real-time weather forecasting system.

## How to Use
1. Clone the repository.
2. Install the required dependencies (`pip install -r requirements.txt`).
3. Run the Jupyter Notebook to preprocess data, train the model, and evaluate results.
4. Use the saved model (`best_tcn_model_step1.h5`) for predictions.

## Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.
