# 🏆 La Liga Match Predictor

An AI-powered machine learning model that predicts La Liga match outcomes using advanced statistical analysis.

## 📊 Current Status

**Last Updated:** 2025-09-30 17:13:27  
**Model Training Accuracy:** 93.1%  
**Overall Prediction Accuracy:** 50.0%  
**Total Matches Predicted:** 20  
**Total Matches with Results:** 10  

## 🔮 Current Gameweek Predictions

### Gameweek 8

⏳ **CA Osasuna vs Getafe CF**  
📅 2025-10-03 19:00  
🎯 **Prediction:** CA Osasuna Win  
**Status:** Awaiting result  
📊 Probabilities: CA Osasuna 48.6% | Draw 31.2% | Getafe CF 20.2%  

⏳ **Real Oviedo vs Levante UD**  
📅 2025-10-04 12:00  
🎯 **Prediction:** Draw  
**Status:** Awaiting result  
📊 Probabilities: Real Oviedo 25.5% | Draw 53.1% | Levante UD 21.4%  

⏳ **Girona FC vs Valencia CF**  
📅 2025-10-04 14:15  
🎯 **Prediction:** Draw  
**Status:** Awaiting result  
📊 Probabilities: Girona FC 26.4% | Draw 50.2% | Valencia CF 23.4%  

⏳ **Athletic Club vs RCD Mallorca**  
📅 2025-10-04 16:30  
🎯 **Prediction:** Athletic Club Win  
**Status:** Awaiting result  
📊 Probabilities: Athletic Club 48.1% | Draw 32.9% | RCD Mallorca 19.0%  

⏳ **Real Madrid CF vs Villarreal CF**  
📅 2025-10-04 19:00  
🎯 **Prediction:** Real Madrid CF Win  
**Status:** Awaiting result  
📊 Probabilities: Real Madrid CF 53.2% | Draw 26.5% | Villarreal CF 20.4%  

⏳ **Deportivo Alavés vs Elche CF**  
📅 2025-10-05 12:00  
🎯 **Prediction:** Draw  
**Status:** Awaiting result  
📊 Probabilities: Deportivo Alavés 17.0% | Draw 42.0% | Elche CF 41.0%  

⏳ **Sevilla FC vs FC Barcelona**  
📅 2025-10-05 14:15  
🎯 **Prediction:** Draw  
**Status:** Awaiting result  
📊 Probabilities: Sevilla FC 15.4% | Draw 44.6% | FC Barcelona 40.0%  

⏳ **RCD Espanyol de Barcelona vs Real Betis Balompié**  
📅 2025-10-05 16:30  
🎯 **Prediction:** Draw  
**Status:** Awaiting result  
📊 Probabilities: RCD Espanyol de Barcelona 24.5% | Draw 40.5% | Real Betis Balompié 35.0%  

⏳ **Real Sociedad de Fútbol vs Rayo Vallecano de Madrid**  
📅 2025-10-05 16:30  
🎯 **Prediction:** Rayo Vallecano de Madrid Win  
**Status:** Awaiting result  
📊 Probabilities: Real Sociedad de Fútbol 32.3% | Draw 28.9% | Rayo Vallecano de Madrid 38.8%  

⏳ **RC Celta de Vigo vs Club Atlético de Madrid**  
📅 2025-10-05 19:00  
🎯 **Prediction:** Club Atlético de Madrid Win  
**Status:** Awaiting result  
📊 Probabilities: RC Celta de Vigo 18.6% | Draw 27.5% | Club Atlético de Madrid 53.9%  

## 📈 Prediction History

### Gameweek 7
*Predicted on: 2025-09-28 10:57:37*

**Final Accuracy: 50.0% (5/10 correct)**

✅ **Girona FC vs RCD Espanyol de Barcelona**  
🎯 Predicted: Draw  
⚽ **Final Result:** Draw (0-0)  
📊 Confidence: 32.6% | 34.9% | 32.4%

✅ **Getafe CF vs Levante UD**  
🎯 Predicted: Draw  
⚽ **Final Result:** Draw (1-1)  
📊 Confidence: 33.3% | 35.4% | 31.3%

❌ **Club Atlético de Madrid vs Real Madrid CF**  
🎯 Predicted: Real Madrid CF Win  
⚽ **Final Result:** Club Atlético de Madrid Win (5-2)  
📊 Confidence: 22.8% | 33.7% | 43.5%

❌ **RCD Mallorca vs Deportivo Alavés**  
🎯 Predicted: Draw  
⚽ **Final Result:** RCD Mallorca Win (1-0)  
📊 Confidence: 28.0% | 46.1% | 25.8%

✅ **Villarreal CF vs Athletic Club**  
🎯 Predicted: Villarreal CF Win  
⚽ **Final Result:** Villarreal CF Win (1-0)  
📊 Confidence: 44.0% | 31.1% | 24.9%

❌ **Rayo Vallecano de Madrid vs Sevilla FC**  
🎯 Predicted: Draw  
⚽ **Final Result:** Sevilla FC Win (0-1)  
📊 Confidence: 39.1% | 39.7% | 21.2%

✅ **Elche CF vs RC Celta de Vigo**  
🎯 Predicted: Elche CF Win  
⚽ **Final Result:** Elche CF Win (2-1)  
📊 Confidence: 59.4% | 29.8% | 10.8%

✅ **FC Barcelona vs Real Sociedad de Fútbol**  
🎯 Predicted: FC Barcelona Win  
⚽ **Final Result:** FC Barcelona Win (2-1)  
📊 Confidence: 75.8% | 17.1% | 7.1%

❌ **Real Betis Balompié vs CA Osasuna**  
🎯 Predicted: CA Osasuna Win  
⚽ **Final Result:** Real Betis Balompié Win (2-0)  
📊 Confidence: 26.9% | 32.9% | 40.2%

❌ **Valencia CF vs Real Oviedo**  
🎯 Predicted: Valencia CF Win  
⚽ **Final Result:** Real Oviedo Win (1-2)  
📊 Confidence: 73.4% | 17.3% | 9.2%

---

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
