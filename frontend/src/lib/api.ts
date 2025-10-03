import { supabase } from "./supabase";
import { Prediction } from "@/types/prediction"; // Assuming Prediction type is defined

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
  const predictions: Prediction[] = data.map((item: any) => ({
    id: item.id,
    gameweek: item.gameweek,
    homeTeam: item.home_team,
    awayTeam: item.away_team,
    predictedResult: item.predicted_result,
    homeProb: item.home_prob,
    awayProb: item.away_prob,
    drawProb: item.draw_prob,
    date: item.match_date,
    actualResult: item.actual_result,
    predictedHomeScore: item.home_score, // NOTE: Assuming your MatchCard logic uses 'predictedHomeScore'
    predictedAwayScore: item.away_score, // NOTE: If these are actual scores for historical data, rename the key in the Prediction type
    actualHomeScore: item.home_score,
    actualAwayScore: item.away_score,
    isCorrect: item.is_correct,
    confidence: Math.max(item.home_prob, item.away_prob, item.draw_prob), // Calculate max confidence for the card
  }));

  return predictions;
}

/**
 * Fetches model statistics from Supabase
 */
export async function fetchModelStats(): Promise<any> {
  const { data, error } = await supabase
    .from("model_stats")
    .select("*")
    .limit(1)
    .single();

  if (error) {
    console.error("Error fetching model stats:", error);
    // You might throw the error or return default stats
    return null;
  }

  return {
    trainingAccuracy: data.training_accuracy,
    currentGameweek: data.current_gameweek,
    lastUpdated: data.last_updated,
  };
}
