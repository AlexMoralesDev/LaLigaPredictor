# La Liga Match Predictor 🔮⚽

A machine learning-powered football match prediction system that analyzes La Liga matches using advanced statistical modeling and Random Forest algorithms.

## 🚀 Features

- **Advanced Statistical Analysis**: Comprehensive team performance metrics including form, attack/defense ratings, and venue-specific performance
- **Machine Learning Predictions**: Random Forest classifier trained on 2+ seasons of historical data
- **Real-time Data**: Fetches live data from football-data.org API
- **Predictive Insights**: Generates probability-based predictions for upcoming matches
- **Quality Opposition Analysis**: Tracks performance against top and bottom-tier teams

## 📊 Current Gameweek Predictions

### Gameweek [CURRENT_GAMEWEEK] Predictions

_Last Updated: [DATE]_

````
============================================================
PREDICTIONS FOR GAMEWEEK 7
============================================================

2025-09-26 19:00
Girona FC vs RCD Espanyol de Barcelona
Prediction: Draw
Probabilities: Girona FC 32.8% | Draw 35.7% | RCD Espanyol de Barcelona 31.5%
Stats: Girona FC PPG: 1.51, Form: 0.40 | RCD Espanyol de Barcelona PPG: 1.20, Form: 1.60

2025-09-27 12:00
Getafe CF vs Levante UD
Prediction: Draw
Probabilities: Getafe CF 33.4% | Draw 35.9% | Levante UD 30.7%
Stats: Getafe CF PPG: 1.16, Form: 1.40 | Levante UD PPG: 0.67, Form: 0.80

2025-09-27 14:15
Club Atlético de Madrid vs Real Madrid CF
Prediction: Real Madrid CF Win
Probabilities: Club Atlético de Madrid 23.2% | Draw 35.4% | Real Madrid CF 41.4%
Stats: Club Atlético de Madrid PPG: 1.96, Form: 1.80 | Real Madrid CF PPG: 2.40, Form: 3.00

2025-09-27 16:30
RCD Mallorca vs Deportivo Alavés
Prediction: Draw
Probabilities: RCD Mallorca 28.8% | Draw 45.7% | Deportivo Alavés 25.5%
Stats: RCD Mallorca PPG: 1.10, Form: 0.40 | Deportivo Alavés PPG: 1.17, Form: 1.00

2025-09-27 19:00
Villarreal CF vs Athletic Club
Prediction: Villarreal CF Win
Probabilities: Villarreal CF 44.0% | Draw 32.4% | Athletic Club 23.6%
Stats: Villarreal CF PPG: 1.66, Form: 2.00 | Athletic Club PPG: 1.80, Form: 1.40

2025-09-28 12:00
Rayo Vallecano de Madrid vs Sevilla FC
Prediction: Draw
Probabilities: Rayo Vallecano de Madrid 37.3`%` | Draw 40.4% | Sevilla FC 22.3%
Stats: Rayo Vallecano de Madrid PPG: 1.16, Form: 0.40 | Sevilla FC PPG: 1.09, Form: 1.40

2025-09-28 14:15
Elche CF vs RC Celta de Vigo
Prediction: Elche CF Win
Probabilities: Elche CF 57.7% | Draw 31.2% | RC Celta de Vigo 11.1%
Stats: Elche CF PPG: 1.67, Form: 1.80 | RC Celta de Vigo PPG: 1.23, Form: 1.00

2025-09-28 16:30
FC Barcelona vs Real Sociedad de Fútbol
Prediction: FC Barcelona Win
Probabilities: FC Barcelona 72.6% | Draw 19.1% | Real Sociedad de Fútbol 8.3%
Stats: FC Barcelona PPG: 2.30, Form: 2.60 | Real Sociedad de Fútbol PPG: 1.35, Form: 0.80

2025-09-28 19:00
Real Betis Balompié vs CA Osasuna
Prediction: CA Osasuna Win
Probabilities: Real Betis Balompié 27.5% | Draw 31.8% | CA Osasuna 40.7%
Stats: Real Betis Balompié PPG: 1.54, Form: 1.60 | CA Osasuna PPG: 1.27, Form: 1.40

2025-09-29 19:00
Valencia CF vs Real Oviedo
Prediction: Valencia CF Win
Probabilities: Valencia CF 71.7% | Draw 20.0% | Real Oviedo 8.3%
Stats: Valencia CF PPG: 1.26, Form: 1.40 | Real Oviedo PPG: 0.50, Form: 0.60
## 🏆 Model Performance

- **Training Accuracy**: ~65-70% on historical data
- **Training Dataset**: 600+ matches from 2023-2025 seasons
- **Features**: 25+ predictive features including form, strength, and situational factors

### Accuracy by Result Type:
- Home Wins: ~68%
- Away Wins: ~62%
- Draws: ~58%

## 🔧 Installation & Setup

### Prerequisites
```bash
pip install requests pandas scikit-learn numpy python-dotenv
````

### Environment Setup

1. Get your API key from [football-data.org](https://www.football-data.org/client/register)
2. Create a `.env` file in the project root:

```
FOOTBALL_API_KEY=your_api_key_here
```

### Running the Predictor

```bash
python la_liga_predictor.py
```

## 📈 Key Features Explained

### Statistical Metrics

- **PPG (Points Per Game)**: Average points earned per match
- **Form Rating**: Weighted average of recent results (W=3, D=1, L=0)
- **Attack/Defense Ratings**: Goals scored/conceded per game
- **Quality Performance**: Results against top-6 and bottom-6 teams

### Prediction Algorithm

The model uses a Random Forest classifier with:

- **300 estimators** for stability
- **Balanced class weights** to handle draws
- **Feature engineering** combining 25+ metrics
- **Cross-validation** on historical data

## 🎯 Feature Importance

Top predictive features:

1. **Overall Strength Difference**: Combined PPG, home advantage, and form
2. **Home Advantage**: Home team's home performance vs away team's away performance
3. **Recent Form**: Last 3-5 match performance trends
4. **Attack vs Defense**: Offensive strength vs defensive weakness matchups
5. **Expected Goals**: Predicted goal output based on team ratings

## 📋 Model Architecture

```
Training Data (2023-2025)
         ↓
Feature Engineering (25+ features)
         ↓
Random Forest Classifier
         ↓
Probability Predictions
         ↓
Match Outcome Predictions
```

## 🔍 Model Limitations

- **Injuries/Suspensions**: Not accounted for in current model
- **Tactical Changes**: Manager/formation changes not tracked
- **Weather Conditions**: Not included in predictions
- **Motivation**: Derby/rivalry factors partially captured
- **Transfer Window**: New signings impact not immediately reflected

## 🚀 Future Enhancements

- [ ] Player-level statistics integration
- [ ] Injury/suspension tracking
- [ ] Weather data incorporation
- [ ] Head-to-head historical records
- [ ] Expected Goals (xG) integration
- [ ] Betting odds comparison
- [ ] Multi-league support

## 📈 Validation & Backtesting

The model is continuously validated against:

- **Historical accuracy** on completed matches
- **Feature importance** analysis
- **Prediction calibration** (predicted vs actual probabilities)
- **Performance by match type** (home/away/neutral)

## 🤝 Contributing

Feel free to contribute improvements:

1. Enhanced feature engineering
2. Alternative ML algorithms
3. Real-time data integration
4. Performance optimizations
5. Additional leagues

## 📄 License

This project is for educational and research purposes. Please comply with football-data.org's API terms of service.

## ⚠️ Disclaimer

**This model is for educational and entertainment purposes only. Sports betting involves risk and should be done responsibly. Past performance does not guarantee future results.**

---

### 📞 Support

For questions or issues, please check:

- API documentation: [football-data.org](https://www.football-data.org/documentation)
- Model methodology in the source code comments
- Feature engineering details in the `create_focused_features()` function

---

**Last Updated**: September 2025  
**Model Version**: 2.1  
**Dataset**: La Liga 2023-2025 seasons
