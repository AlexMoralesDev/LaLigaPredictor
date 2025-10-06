import { Prediction } from "@/types/prediction";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2, XCircle } from "lucide-react";
import { getTeamLogo } from "@/lib/teamLogos";
import { useState } from "react";

interface MatchCardProps {
  prediction: Prediction;
  showActual?: boolean;
}

const MatchCard = ({ prediction, showActual = false }: MatchCardProps) => {
  const [homeLogoError, setHomeLogoError] = useState(false);
  const [awayLogoError, setAwayLogoError] = useState(false);

  // Get logo URLs with error logging
  const homeLogoUrl = getTeamLogo(prediction.homeTeam);
  const awayLogoUrl = getTeamLogo(prediction.awayTeam);

  // Log for debugging
  console.log("Home Team:", prediction.homeTeam, "Logo URL:", homeLogoUrl);
  console.log("Away Team:", prediction.awayTeam, "Logo URL:", awayLogoUrl);

  // Determine colors based on probability ranking
  const getProbabilityColors = () => {
    const probs = [
      { name: "home", value: prediction.homeWinProbability, label: "Home" },
      { name: "draw", value: prediction.drawProbability, label: "Draw" },
      { name: "away", value: prediction.awayWinProbability, label: "Away" },
    ];

    // Sort by probability (highest to lowest)
    probs.sort((a, b) => b.value - a.value);

    // Assign colors:
    const colorMap: Record<string, string> = {};
    colorMap[probs[0].name] = "bg-[#5DC169]"; // Highest - Green
    colorMap[probs[1].name] = "bg-[#DEB94C]"; // Middle - Yellow
    colorMap[probs[2].name] = "bg-[#813835]"; // Lowest - Red

    return {
      homeColor: colorMap["home"],
      drawColor: colorMap["draw"],
      awayColor: colorMap["away"],
    };
  };

  const colors = getProbabilityColors();

  const isPredictionCorrect = () => {
    // Return null if not showing actual results or if actual scores don't exist
    if (
      !showActual ||
      prediction.actualHomeScore === null ||
      prediction.actualHomeScore === undefined
    )
      return null;
    if (
      prediction.actualAwayScore === null ||
      prediction.actualAwayScore === undefined
    )
      return null;

    // Use the isCorrect field from the database if available
    if (typeof prediction.isCorrect === "boolean") {
      return prediction.isCorrect;
    }

    // Fallback: Calculate correctness based on predicted vs actual result
    const predictedResult =
      prediction.predictedHomeScore > prediction.predictedAwayScore
        ? "home"
        : prediction.predictedHomeScore < prediction.predictedAwayScore
          ? "away"
          : "draw";

    const actualResult =
      prediction.actualHomeScore > prediction.actualAwayScore
        ? "home"
        : prediction.actualHomeScore < prediction.actualAwayScore
          ? "away"
          : "draw";

    return predictedResult === actualResult;
  };

  const accuracy = isPredictionCorrect();

  // Convert to user's local timezone
  const matchDate = new Date(prediction.date).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short",
  });

  // Fallback logo generator
  const generateFallbackLogo = (teamName: string) => {
    const initials = teamName
      .split(" ")
      .map((word) => word.charAt(0))
      .join("")
      .substring(0, 2)
      .toUpperCase();
    return `https://ui-avatars.com/api/?name=${encodeURIComponent(initials)}&background=1e293b&color=fff&size=128&bold=true&font-size=0.5`;
  };

  const handleHomeLogoError = (e: React.SyntheticEvent<HTMLImageElement>) => {
    console.error(`Failed to load logo for ${prediction.homeTeam}`);
    console.error("Attempted URL:", homeLogoUrl);
    setHomeLogoError(true);
    e.currentTarget.src = generateFallbackLogo(prediction.homeTeam);
  };

  const handleAwayLogoError = (e: React.SyntheticEvent<HTMLImageElement>) => {
    console.error(`Failed to load logo for ${prediction.awayTeam}`);
    console.error("Attempted URL:", awayLogoUrl);
    setAwayLogoError(true);
    e.currentTarget.src = generateFallbackLogo(prediction.awayTeam);
  };

  return (
    <Card className="p-4 sm:p-6 bg-gradient-card border-border hover:border-primary/50 transition-all duration-300 hover:shadow-lg hover:shadow-primary/10">
      <div className="flex items-center justify-between mb-4">
        <Badge variant="secondary" className="text-xs">
          GW {prediction.gameweek}
        </Badge>
        <span className="text-xs text-muted-foreground">{matchDate}</span>
      </div>

      <div className="flex items-center justify-center gap-2 sm:gap-6 md:gap-8">
        {/* Home Team */}
        <div className="flex-1 flex items-center justify-end gap-2 sm:gap-3 min-w-0 max-w-[40%]">
          <p className="font-semibold text-foreground text-xs sm:text-sm md:text-base lg:text-lg text-right break-words">
            {prediction.homeTeam}
          </p>
          <div className="w-10 h-10 sm:w-12 sm:h-12 flex-shrink-0 flex items-center justify-center bg-white/5 rounded-lg p-1 relative">
            <img
              src={homeLogoUrl}
              alt={`${prediction.homeTeam} logo`}
              className="w-full h-full object-contain"
              loading="lazy"
              onError={handleHomeLogoError}
              crossOrigin="anonymous"
            />
            {homeLogoError && (
              <div className="absolute inset-0 flex items-center justify-center text-xs text-red-500 bg-slate-800/50 rounded-lg">
                ⚠️
              </div>
            )}
          </div>
        </div>

        {/* Score/VS Divider */}
        <div className="flex flex-col items-center gap-2 flex-shrink-0">
          {showActual &&
          prediction.actualHomeScore !== null &&
          prediction.actualHomeScore !== undefined &&
          prediction.actualAwayScore !== null &&
          prediction.actualAwayScore !== undefined ? (
            // Show actual score when match is finished (History tab)
            <>
              <div className="flex items-center gap-2 sm:gap-3">
                <span className="text-2xl sm:text-3xl font-bold text-foreground">
                  {prediction.actualHomeScore}
                </span>
                <span className="text-lg sm:text-xl text-muted-foreground">
                  -
                </span>
                <span className="text-2xl sm:text-3xl font-bold text-foreground">
                  {prediction.actualAwayScore}
                </span>
              </div>
              <span className="text-xs text-muted-foreground">Final Score</span>
              {accuracy !== null && (
                <div className="mt-1">
                  {accuracy ? (
                    <CheckCircle2 className="w-5 h-5 text-success" />
                  ) : (
                    <XCircle className="w-5 h-5 text-destructive" />
                  )}
                </div>
              )}
            </>
          ) : (
            // Show VS when match hasn't been played (Current Predictions tab)
            <span className="text-xs font-medium text-muted-foreground">
              VS
            </span>
          )}
        </div>

        {/* Away Team */}
        <div className="flex-1 flex items-center justify-start gap-2 sm:gap-3 min-w-0 max-w-[40%]">
          <div className="w-10 h-10 sm:w-12 sm:h-12 flex-shrink-0 flex items-center justify-center bg-white/5 rounded-lg p-1 relative">
            <img
              src={awayLogoUrl}
              alt={`${prediction.awayTeam} logo`}
              className="w-full h-full object-contain"
              loading="lazy"
              onError={handleAwayLogoError}
              crossOrigin="anonymous"
            />
            {awayLogoError && (
              <div className="absolute inset-0 flex items-center justify-center text-xs text-red-500 bg-slate-800/50 rounded-lg">
                ⚠️
              </div>
            )}
          </div>
          <p className="font-semibold text-foreground text-xs sm:text-sm md:text-base lg:text-lg text-left break-words">
            {prediction.awayTeam}
          </p>
        </div>
      </div>

      {/* Probability bar is always shown */}

      {/* Segmented Probability Bar */}
      <div className="mt-4 pt-4 border-t border-border">
        <div className="flex items-center justify-between mb-2 text-[9px] sm:text-[10px] md:text-xs gap-1">
          <span className="text-muted-foreground flex-shrink-0">
            <span className="font-semibold text-foreground">
              {prediction.homeWinProbability}%
            </span>{" "}
            <span className="hidden sm:inline">Home</span>
            <span className="sm:hidden">H</span>
          </span>
          <span className="text-muted-foreground flex-shrink-0">
            <span className="font-semibold text-foreground">
              {prediction.drawProbability}%
            </span>{" "}
            <span className="hidden sm:inline">Draw</span>
            <span className="sm:hidden">D</span>
          </span>
          <span className="text-muted-foreground flex-shrink-0">
            <span className="font-semibold text-foreground">
              {prediction.awayWinProbability}%
            </span>{" "}
            <span className="hidden sm:inline">Away</span>
            <span className="sm:hidden">A</span>
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
