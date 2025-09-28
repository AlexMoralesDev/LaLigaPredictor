# 🏆 La Liga Match Predictor

An AI-powered machine learning model that predicts La Liga match outcomes using advanced statistical analysis.

## 📊 Current Status

**Last Updated:** 2025-09-28 10:57:37  
**Model Training Accuracy:** 93.4%  
**Overall Prediction Accuracy:** 42.9%  
**Total Matches Predicted:** 10  
**Total Matches with Results:** 7  

## 🔮 Current Gameweek Predictions

### Gameweek 7

✅ **Girona FC vs RCD Espanyol de Barcelona**  
📅 2025-09-26 19:00  
🎯 **Prediction:** Draw  
**Result:** Draw (0-0) - ✅ CORRECT  
📊 Probabilities: Girona FC 32.6% | Draw 34.9% | RCD Espanyol de Barcelona 32.4%  

✅ **Getafe CF vs Levante UD**  
📅 2025-09-27 12:00  
🎯 **Prediction:** Draw  
**Result:** Draw (1-1) - ✅ CORRECT  
📊 Probabilities: Getafe CF 33.3% | Draw 35.4% | Levante UD 31.3%  

❌ **Club Atlético de Madrid vs Real Madrid CF**  
📅 2025-09-27 14:15  
🎯 **Prediction:** Real Madrid CF Win  
**Result:** Club Atlético de Madrid Win (5-2) - ❌ WRONG  
📊 Probabilities: Club Atlético de Madrid 22.8% | Draw 33.7% | Real Madrid CF 43.5%  

❌ **RCD Mallorca vs Deportivo Alavés**  
📅 2025-09-27 16:30  
🎯 **Prediction:** Draw  
**Result:** RCD Mallorca Win (1-0) - ❌ WRONG  
📊 Probabilities: RCD Mallorca 28.0% | Draw 46.1% | Deportivo Alavés 25.8%  

✅ **Villarreal CF vs Athletic Club**  
📅 2025-09-27 19:00  
🎯 **Prediction:** Villarreal CF Win  
**Result:** Villarreal CF Win (1-0) - ✅ CORRECT  
📊 Probabilities: Villarreal CF 44.0% | Draw 31.1% | Athletic Club 24.9%  

❌ **Rayo Vallecano de Madrid vs Sevilla FC**  
📅 2025-09-28 12:00  
🎯 **Prediction:** Draw  
**Result:** Sevilla FC Win (0-1) - ❌ WRONG  
📊 Probabilities: Rayo Vallecano de Madrid 39.1% | Draw 39.7% | Sevilla FC 21.2%  

❌ **Elche CF vs RC Celta de Vigo**  
📅 2025-09-28 14:15  
🎯 **Prediction:** Elche CF Win  
**Result:** Draw (1-1) - ❌ WRONG  
📊 Probabilities: Elche CF 59.4% | Draw 29.8% | RC Celta de Vigo 10.8%  

⏳ **FC Barcelona vs Real Sociedad de Fútbol**  
📅 2025-09-28 16:30  
🎯 **Prediction:** FC Barcelona Win  
**Status:** Awaiting result  
📊 Probabilities: FC Barcelona 75.8% | Draw 17.1% | Real Sociedad de Fútbol 7.1%  

⏳ **Real Betis Balompié vs CA Osasuna**  
📅 2025-09-28 19:00  
🎯 **Prediction:** CA Osasuna Win  
**Status:** Awaiting result  
📊 Probabilities: Real Betis Balompié 26.9% | Draw 32.9% | CA Osasuna 40.2%  

⏳ **Valencia CF vs Real Oviedo**  
📅 2025-09-29 19:00  
🎯 **Prediction:** Valencia CF Win  
**Status:** Awaiting result  
📊 Probabilities: Valencia CF 73.4% | Draw 17.3% | Real Oviedo 9.2%  

## 📈 Prediction History

No historical predictions available yet. Predictions will appear here after gameweeks are completed.

## 🤖 Model Information

### Features Used
- Points per game difference
- Goal difference per game
- Home/Away venue-specific performance
- Attack vs Defense matchup analysis
- Recent form (last 3 and 5 matches)
- Performance against top/bottom teams
- Clean sheet rates and defensive metrics
- Expected goals calculations

### Algorithm
- **Random Forest Classifier** with 300 trees
- Trained on 2+ seasons of La Liga data
- Features engineered for maximum predictive power
- Handles class imbalance with balanced weights

## 📋 How to Use

1. Clone this repository
2. Set up your Football-Data.org API key in `.env` file
3. Run `python predictor.py` to generate new predictions
4. Check this README for the latest predictions and results

## 📊 Accuracy Breakdown

The model's accuracy is tracked across different result types:
- **Home Wins**: Historically strong performance
- **Away Wins**: Moderate accuracy  
- **Draws**: Most challenging to predict (as expected)

---

*Predictions are for entertainment purposes only. Past performance does not guarantee future results.*
