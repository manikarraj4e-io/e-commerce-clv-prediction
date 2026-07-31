import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
from main import model, X  # Imports the trained model and features from your main script

# Plot feature importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=10, height=0.5, color='teal')
plt.title("Top Features Driving Customer Lifetime Value (CLV)")
plt.savefig("feature_importance.png", bbox_inches='tight')
plt.show()
print("Feature importance chart saved as 'feature_importance.png'!")