import { Prediction } from "@/types/prediction";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2, XCircle, Minus } from "lucide-react";

interface MatchCardProps {
  prediction: Prediction;
  showActual?: boolean;
}

const MatchCard = ({ prediction, showActual = false }: MatchCardProps) => {
  const isPredictionCorrect = () => {
    if (!showActual || prediction.actualHomeScore === undefined) return null;
    
    const predictedResult = prediction.predictedHomeScore > prediction.predictedAwayScore ? "home" :
                           prediction.predictedHomeScore < prediction.predictedAwayScore ? "away" : "draw";
    const actualResult = prediction.actualHomeScore > prediction.actualAwayScore! ? "home" :
                        prediction.actualHomeScore < prediction.actualAwayScore! ? "away" : "draw";
    
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

      <div className="flex items-center justify-between gap-4">
        {/* Home Team */}
        <div className="flex-1 text-right">
          <p className="font-semibold text-foreground mb-2">{prediction.homeTeam}</p>
          <p className="text-3xl font-bold text-primary">{prediction.predictedHomeScore}</p>
          {showActual && prediction.actualHomeScore !== undefined && (
            <p className="text-sm text-muted-foreground mt-1">
              Actual: {prediction.actualHomeScore}
            </p>
          )}
        </div>

        {/* VS Divider */}
        <div className="flex flex-col items-center gap-2">
          <span className="text-xs font-medium text-muted-foreground">VS</span>
          {accuracy !== null && (
            <div className="mt-2">
              {accuracy ? (
                <CheckCircle2 className="w-5 h-5 text-success" />
              ) : (
                <XCircle className="w-5 h-5 text-destructive" />
              )}
            </div>
          )}
        </div>

        {/* Away Team */}
        <div className="flex-1 text-left">
          <p className="font-semibold text-foreground mb-2">{prediction.awayTeam}</p>
          <p className="text-3xl font-bold text-primary">{prediction.predictedAwayScore}</p>
          {showActual && prediction.actualAwayScore !== undefined && (
            <p className="text-sm text-muted-foreground mt-1">
              Actual: {prediction.actualAwayScore}
            </p>
          )}
        </div>
      </div>

      {/* Confidence Bar */}
      <div className="mt-4 pt-4 border-t border-border">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-muted-foreground">Confidence</span>
          <span className="text-xs font-semibold text-foreground">{prediction.confidence}%</span>
        </div>
        <div className="w-full bg-secondary rounded-full h-2 overflow-hidden">
          <div
            className="bg-gradient-pitch h-full transition-all duration-500 rounded-full"
            style={{ width: `${prediction.confidence}%` }}
          />
        </div>
      </div>
    </Card>
  );
};

export default MatchCard;
