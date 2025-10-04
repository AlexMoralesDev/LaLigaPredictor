import { supabase } from "./supabase";
import { Prediction } from "@/types/prediction";

/**
 * Fetches all historical and current predictions from Supabase
 * @returns An array of Prediction objects
 */
export async function fetchAllPredictions(): Promise<Prediction[]> {
  const { data, error } = await supabase
    .from("predictions")
    .select("*")
    .order("match_date", { ascending: true });

  if (error) {
    console.error("Error fetching predictions:", error);
    throw error;
  }

  // Map the Supabase data to your Prediction type
  const predictions: Prediction[] = data.map((item: any) => {
    // Parse predicted_result if it's a score like "2-1"
    let predictedHomeScore = 0;
    let predictedAwayScore = 0;

    if (item.predicted_result && item.predicted_result.includes("-")) {
      [predictedHomeScore, predictedAwayScore] = item.predicted_result
        .split("-")
        .map(Number);
    }

    return {
      id: item.id,
      gameweek: item.gameweek,
      homeTeam: item.home_team,
      awayTeam: item.away_team,
      predictedResult: item.predicted_result,
      homeWinProbability: parseFloat(item.home_prob) || 0,
      awayWinProbability: parseFloat(item.away_prob) || 0,
      drawProbability: parseFloat(item.draw_prob) || 0,
      date: item.match_date,
      actualResult: item.actual_result,
      actualHomeScore: item.home_score,
      actualAwayScore: item.away_score,
      predictedHomeScore: predictedHomeScore,
      predictedAwayScore: predictedAwayScore,
      isCorrect: item.is_correct,
      predictedAt: item.predicted_at,
      createdAt: item.created_at,
      confidence: Math.max(
        parseFloat(item.home_prob) || 0,
        parseFloat(item.away_prob) || 0,
        parseFloat(item.draw_prob) || 0,
      ),
    };
  });

  return predictions;
}

/**
 * Fetches model statistics from Supabase
 */
export async function fetchModelStats(): Promise<any> {
  const { data, error } = await supabase
    .from("model_stats")
    .select("*")
    .order("last_updated", { ascending: false })
    .limit(1)
    .single();

  if (error) {
    console.error("Error fetching model stats:", error);
    return null;
  }

  return {
    trainingAccuracy: data.training_accuracy,
    currentGameweek: data.current_gameweek,
    lastUpdated: data.last_updated,
  };
}
