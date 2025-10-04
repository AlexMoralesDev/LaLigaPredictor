import { Prediction } from "@/types/prediction";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2, XCircle } from "lucide-react";

interface MatchCardProps {
  prediction: Prediction;
  showActual?: boolean;
}

const MatchCard = ({ prediction, showActual = false }: MatchCardProps) => {
  // Determine colors based on probability ranking
  const getProbabilityColors = () => {
    const probs = [
      { name: "home", value: prediction.homeWinProbability, label: "Home" },
      { name: "draw", value: prediction.drawProbability, label: "Draw" },
      { name: "away", value: prediction.awayWinProbability, label: "Away" },
    ];

    // Sort by probability (highest to lowest)
    probs.sort((a, b) => b.value - a.value);

    // Assign colors: highest = green, middle = yellow, lowest = red
    const colorMap: Record<string, string> = {};
    colorMap[probs[0].name] = "bg-green-500"; // Highest
    colorMap[probs[1].name] = "bg-yellow-500"; // Middle
    colorMap[probs[2].name] = "bg-red-500"; // Lowest

    return {
      homeColor: colorMap["home"],
      drawColor: colorMap["draw"],
      awayColor: colorMap["away"],
    };
  };

  const colors = getProbabilityColors();

  const isPredictionCorrect = () => {
    if (!showActual || prediction.actualHomeScore === undefined) return null;

    const predictedResult =
      prediction.predictedHomeScore > prediction.predictedAwayScore
        ? "home"
        : prediction.predictedHomeScore < prediction.predictedAwayScore
          ? "away"
          : "draw";

    const actualResult =
      prediction.actualHomeScore > prediction.actualAwayScore!
        ? "home"
        : prediction.actualHomeScore < prediction.actualAwayScore!
          ? "away"
          : "draw";

    return predictedResult === actualResult;
  };

  const accuracy = isPredictionCorrect();

  const matchDate = new Date(prediction.date).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <Card className="p-6 bg-gradient-card border-border hover:border-primary/50 transition-all duration-300 hover:shadow-lg hover:shadow-primary/10">
      <div className="flex items-center justify-between mb-4">
        <Badge variant="secondary" className="text-xs">
          GW {prediction.gameweek}
        </Badge>
        <span className="text-xs text-muted-foreground">{matchDate}</span>
      </div>

      <div className="flex items-center justify-center gap-8">
        {/* Home Team */}
        <div className="flex-1 text-center">
          <p className="font-semibold text-foreground text-lg">
            {prediction.homeTeam}
          </p>
        </div>

        {/* VS Divider */}
        <div className="flex flex-col items-center gap-2">
          <span className="text-xs font-medium text-muted-foreground">VS</span>
          {accuracy !== null && (
            <div className="mt-1">
              {accuracy ? (
                <CheckCircle2 className="w-5 h-5 text-success" />
              ) : (
                <XCircle className="w-5 h-5 text-destructive" />
              )}
            </div>
          )}
        </div>

        {/* Away Team */}
        <div className="flex-1 text-center">
          <p className="font-semibold text-foreground text-lg">
            {prediction.awayTeam}
          </p>
        </div>
      </div>

      {showActual &&
        (prediction.actualHomeScore !== undefined ||
          prediction.actualAwayScore !== undefined) && (
          <div className="text-center mt-4 pt-4 border-t border-border">
            <p className="text-sm text-muted-foreground">
              Actual Result:{" "}
              <span className="font-semibold text-foreground">
                {prediction.actualResult ||
                  `${prediction.actualHomeScore} - ${prediction.actualAwayScore}`}
              </span>
            </p>
          </div>
        )}

      {/* Segmented Probability Bar */}
      <div className="mt-4 pt-4 border-t border-border">
        <div className="flex items-center justify-between mb-2 text-xs">
          <span className="text-muted-foreground">
            <span className="font-semibold text-foreground">
              {prediction.homeWinProbability}%
            </span>{" "}
            Home
          </span>
          <span className="text-muted-foreground">
            <span className="font-semibold text-foreground">
              {prediction.drawProbability}%
            </span>{" "}
            Draw
          </span>
          <span className="text-muted-foreground">
            <span className="font-semibold text-foreground">
              {prediction.awayWinProbability}%
            </span>{" "}
            Away
          </span>
        </div>
        <div className="w-full bg-secondary rounded-full h-3 overflow-hidden flex">
          <div
            className={`${colors.homeColor} h-full transition-all duration-500`}
            style={{ width: `${prediction.homeWinProbability}%` }}
          />
          <div
            className={`${colors.drawColor} h-full transition-all duration-500`}
            style={{ width: `${prediction.drawProbability}%` }}
          />
          <div
            className={`${colors.awayColor} h-full transition-all duration-500`}
            style={{ width: `${prediction.awayWinProbability}%` }}
          />
        </div>
      </div>
    </Card>
  );
};

export default MatchCard;
