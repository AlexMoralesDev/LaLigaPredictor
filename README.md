# La Liga Match Predictor

**Live Application:** [https://laligapredictor.netlify.app](https://laligapredictor.netlify.app)
![La Liga Predictor Demo](GitVisuals/LaLigaDemo.gif)

A full-stack machine learning web app that predicts Spanish La Liga match outcomes using statistical modeling and real-time data.  
Developed with **Python** for the backend prediction engine and **React (TypeScript)** for the interactive frontend interface.

---

## Project Overview

This project demonstrates an end-to-end workflow for a football match prediction system — from model training and data processing to frontend deployment.  
The app generates outcome probabilities for each match (Home Win, Draw, Away Win) and provides a clean interface for exploring predictions and accuracy trends.

---

## Key Features

- **Prediction Engine:** Random Forest model for real-time outcome predictions
- **Interactive UI:** Responsive web app built with React and Tailwind CSS
- **Automation:** Scheduled prediction updates for each gameweek
- **Performance Tracking:** Historical accuracy and confidence metrics
- **Deployment:** Fully automated CI/CD pipeline with Netlify

---

## Technical Overview

### Machine Learning

- **Model:** Random Forest Classifier (300 trees, tuned hyperparameters)
- **Features:** Team form, goal statistics, venue performance, head-to-head history
- **Evaluation:** Cross-validation and feature importance tracking
- **Files:**
  - `predictor.py` – main model pipeline and prediction logic
  - `predictor_dev.py` – experimental version for testing new features

### Frontend

- Built with **React 18** and **TypeScript**
- Styled using **Tailwind CSS** and **shadcn/ui**
- Data visualization with clean, responsive layouts
- Deployed via **Netlify** with continuous deployment

---

## How to Run Locally

1. **Clone the repository**

   ```bash
   git clone https://github.com/AlexMoralesDev/LaLigaMatchPredictor.git
   cd LaLigaMatchPredictor
   ```

2. **Backend setup**

   ```bash
   pip install -r requirements.txt
   python predictor.py
   ```

3. **Frontend setup**

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

4. Open your browser at **http://localhost:5173**

---

## Project Structure

```
LaLigaMatchPredictor/
├── backend/
│   ├── predictor.py        # Main ML model and prediction logic
│   └── predictor_dev.py    # Development/testing version
├── frontend/               # React + TypeScript web app
└── README.md
```

---

## Future Improvements

- Add player-level and referee data
- Expand coverage to other European leagues
- Improve visual analytics and mobile responsiveness
- Explore ensemble or deep learning methods

---

**Author:** [Alex Morales Trevisan](https://www.linkedin.com/in/alex-morales-dev/)  
**GitHub:** [github.com/AlexMoralesDev](https://github.com/AlexMoralesDev)  
**LinkedIn:** [https://www.linkedin.com/in/alex-morales-dev/](https://www.linkedin.com/in/alex-morales-dev/)  
**Last Updated:** October 2025
