export interface Prediction {
  id: string;
  gameweek: number;
  homeTeam: string;
  awayTeam: string;
  predictedHomeScore: number;
  predictedAwayScore: number;
  confidence: number;
  actualHomeScore?: number;
  actualAwayScore?: number;
  date: string;
}

export interface GameweekStats {
  gameweek: number;
  totalMatches: number;
  correctPredictions: number;
  accuracy: number;
}
