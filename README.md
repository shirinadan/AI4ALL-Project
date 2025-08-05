# Bizlens: Startup Success Evaluation

> A machine learning project to predict early-stage startup success scores based on funding, founding data, and industry insights.

---

## 📌 Project Overview

Bizlens is an ML-powered tool that estimates the success potential of startups using a regression model trained on global startup investment data. The goal is to assist entrepreneurs in evaluating their business positioning through explainable, data-driven predictions.

This project was developed as part of an AI/ML learning program focused on building real-world, responsible AI systems.

---

## ⚙️ Tech Stack

- **Python 3.11+**
- **Pandas**, **NumPy** – data preprocessing  
- **Scikit-learn** – model development (Random Forest)  
- **Matplotlib**, **Seaborn** – EDA and visualizations
- **Flask** - Back-end fetching data for model
- **React** - Front-end UI for web app

---

## 🗂️ Dataset

- **Source**: Startup Investments (Crunchbase) on kaggle
- **Size**: ~9,000 entries  
- **Features**: 30 total → reduced to 15 through correlation analysis  
- **Filtered**: Startups founded latest 2014  
- **Label**: Custom success score (details below)

---

## 🔄 Label Design

We initially received a “success score” from the dataset, but:

- It was noisy and lacked transparency  
- We attempted to **engineer our own success formula**, but the model just memorized it  
- Final approach:
  - Defined success using 2–3 key features (e.g., total funding, acquisition status)  
  - **Removed those features from training** to prevent data leakage  
  - Result: The model infers success based on **other patterns**, not direct labels

---

## 📊 Model Details

| Component           | Value                          |
|--------------------|---------------------------------|
| Model Type          | Supervised Regression           |
| Algorithm           | RandomForestRegressor           |
| Train/Test Split    | 80/20                           |
| R² Score (Test)     | 0.6826                          |
| RMSE                | 0.0565                          |
| MAE                 | 0.0380                          |

---

## 🔍 Feature Importance

Top predictive features include:
- Number of funding rounds  
- Industry category  
- Founding year  
- Team size estimates

*We used feature importance plots to identify top drivers of success predictions.*

---

## 🧪 Ethical & Fairness Considerations

- **Survivorship Bias**: Most failed startups are excluded from the dataset  
- **Timeline Gaps**: Post-2014 startups are not included
- **Nuanced Success Factors**: Our definition of “success” is based on a few features and may vary across regions or industries.

---

## 📈 Future Work

- 📡 Add updated and post-2014 data  
- 💬 Incorporate qualitative factors (e.g., founder background, pitch tone)  
- 🔍 Evaluate fairness across gender, region, and underrepresented founder groups

---

## 👥 Contributors

- **Ebyan Jama** – University of Minnesota  
- **Elisa Yu** – Tufts University  
- **Sarah Toussaint** – NYU
- **Victor Olivo** - Rutgers University 
- **Shirina Daniels** – Florida International University

---

## 📚 References

- [Startup Statistics 2025 – DemandSage](https://www.demandsage.com/startup-statistics/)  
- [AI-native Startups on the Rise – Investment Monitor](https://www.investmentmonitor.ai/news/global-start-up-ecosystem-value-down-but-ai-native-start-ups-on-the-rise-report/)  
- [Entrepreneurship in the U.S. – American Progress](https://www.americanprogress.org/article/entrepreneurship-startups-and-business-formation-are-booming-across-the-u-s/)

---

## ⚠️ License

This project is for educational and experimental purposes only. It is **not production-ready** and should not be used to automate real investment decisions without human oversight.

---
