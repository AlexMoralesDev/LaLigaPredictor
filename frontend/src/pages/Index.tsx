import { useState, useEffect, useMemo } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import MatchCard from "@/components/MatchCard";
import StatsCard from "@/components/StatsCard";
import { Prediction } from "@/types/prediction";
import { Trophy, Target, TrendingUp } from "lucide-react";
import { fetchAllPredictions, fetchModelStats } from "@/lib/api"; // ⬅️ IMPORT API FUNCTIONS

// The correct, singular declaration of the isCorrect helper function
const isCorrect = (prediction: Prediction) => {
  // Check if actual scores exist (i.e., the match is complete)
  if (
    prediction.actualHomeScore === undefined ||
    prediction.actualAwayScore === undefined ||
    prediction.actualHomeScore === null ||
    prediction.actualAwayScore === null
  ) {
    return false;
  }

  // Determine predicted result based on *predicted* scores (or implied scores from the model)
  // NOTE: This assumes that for current predictions, the model output is used, but for historical data
  // the model only saved 'home_score'/'away_score' when the match was complete.
  // If the MatchCard relies on actual scores for historical accuracy, we need to ensure this logic is sound.
  // Given the Python script saves the *actual* scores to home_score/away_score for completed matches,
  // the comparison should use the actual result vs the predicted string result.

  // The Python script saves a boolean `is_correct` field. It's much simpler to use that.
  if (typeof prediction.isCorrect === "boolean") {
    return prediction.isCorrect;
  }

  // Fallback logic if isCorrect field is missing/null, comparing the predicted string result with the actual result.
  // 1. Determine actual match outcome
  const actualResult =
    prediction.actualHomeScore > prediction.actualAwayScore!
      ? "HOME_TEAM"
      : prediction.actualHomeScore < prediction.actualAwayScore!
        ? "AWAY_TEAM"
        : "DRAW";

  // 2. Map predicted string to match the actual result for comparison
  const predictedResult = prediction.predictedResult.includes(
    prediction.homeTeam,
  )
    ? "HOME_TEAM"
    : prediction.predictedResult.includes(prediction.awayTeam)
      ? "AWAY_TEAM"
      : "DRAW";

  return predictedResult === actualResult;
};
console.log("Index component rendering started.");

const Index = () => {
  // ...const Index = () => {
  const [selectedTeam, setSelectedTeam] = useState("All Teams");
  const [sortBy, setSortBy] = useState("date");
  const [allPredictions, setAllPredictions] = useState<Prediction[]>([]);
  const [modelStats, setModelStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  const laLigaTeams = useMemo(() => {
    const teams = new Set<string>();
    allPredictions.forEach((p) => {
      teams.add(p.homeTeam);
      teams.add(p.awayTeam);
    });
    return ["All Teams", ...Array.from(teams).sort()];
  }, [allPredictions]);

  // 1. Fetch data on component mount
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        const [predictionsData, statsData] = await Promise.all([
          fetchAllPredictions(),
          fetchModelStats(),
        ]);
        setAllPredictions(predictionsData);
        setModelStats(statsData);
      } catch (e) {
        console.error("Failed to load all data", e);
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, []);

  // 2. Separate current and historical predictions
  const { currentPredictions, historicalPredictions } = useMemo(() => {
    const current: Prediction[] = [];
    const historical: Prediction[] = [];

    const currentGW = modelStats?.currentGameweek;

    allPredictions.forEach((p) => {
      // The Python script saves 'home_score'/'away_score' ONLY for completed matches,
      // and sets 'gameweek' to the current gameweek for both current and completed matches in that week.

      // Match is complete if actualHomeScore (mapped from home_score) is not null.
      if (p.actualHomeScore !== null) {
        historical.push(p);
      } else if (p.gameweek === currentGW && p.actualHomeScore === null) {
        // Match is part of the current gameweek and has no score yet (a true prediction)
        current.push(p);
      }
      // Note: Matches from past gameweeks that haven't been completed will be ignored,
      // which is usually correct for a live prediction dashboard.
    });

    return {
      currentPredictions: current,
      // Sort historical by date descending for better viewing
      historicalPredictions: historical.sort(
        (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime(),
      ),
    };
  }, [allPredictions, modelStats]);

  // 3. Update calculateStats to use fetched data
  const calculateStats = () => {
    const withActuals = historicalPredictions.filter(
      (p) => p.actualHomeScore !== null && p.actualHomeScore !== undefined,
    );
    // Use the `isCorrect` field from the database if available, otherwise use the function logic.
    const correct = withActuals.filter(isCorrect).length;
    const accuracy =
      withActuals.length > 0
        ? ((correct / withActuals.length) * 100).toFixed(1)
        : "0";

    const historicalConfidence = historicalPredictions
      .map((p) => p.confidence)
      .filter((c) => c !== undefined && c !== null);
    const avgConfidence =
      historicalConfidence.length > 0
        ? (
            historicalConfidence.reduce((sum, c) => sum + c, 0) /
            historicalConfidence.length
          ).toFixed(1)
        : "0";

    return {
      totalPredictions: withActuals.length,
      correctPredictions: correct,
      accuracy,
      avgConfidence,
      trainingAccuracy: modelStats?.trainingAccuracy
        ? parseFloat(modelStats.trainingAccuracy).toFixed(1)
        : "-",
      currentGameweek: modelStats?.currentGameweek || "-",
    };
  };

  const stats = calculateStats();

  const filterPredictions = (predictions: Prediction[]) => {
    let filtered = [...predictions];

    if (selectedTeam !== "All Teams") {
      filtered = filtered.filter(
        (p) => p.homeTeam === selectedTeam || p.awayTeam === selectedTeam,
      );
    }

    if (sortBy === "confidence") {
      filtered.sort((a, b) => b.confidence - a.confidence);
    } else if (sortBy === "accuracy") {
      // Sorting by accuracy is only meaningful for historical data, where isCorrect returns a boolean
      filtered.sort((a, b) => {
        const aCorrect = isCorrect(a);
        const bCorrect = isCorrect(b);
        return (bCorrect ? 1 : 0) - (aCorrect ? 1 : 0);
      });
    } else {
      // Default sort by date ascending for current, or reverse for history (already handled in useMemo)
      filtered.sort(
        (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime(),
      );
    }

    return filtered;
  };

  // 4. Handle Loading State
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center text-xl">
        Loading Predictions...
      </div>
    );
  }

  // 5. Render component
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      {/* ... existing header code ... */}

      <div className="container mx-auto px-4 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <StatsCard
            title="Historical Accuracy"
            value={`${stats.accuracy}%`}
            subtitle={`${stats.correctPredictions}/${stats.totalPredictions} correct`}
            trend="up"
          />
          <StatsCard
            title="Avg Confidence"
            value={`${stats.avgConfidence}%`}
            subtitle={`Model Training: ${stats.trainingAccuracy}%`}
          />
          <StatsCard
            title="Current Gameweek"
            value={stats.currentGameweek}
            subtitle={`${currentPredictions.length} matches to be played`}
          />
          <StatsCard
            title="Total Matches"
            value={historicalPredictions.length + currentPredictions.length}
            subtitle="Predicted so far"
          />
        </div>

        {/* Filters */}
        <div className="flex flex-col sm:flex-row gap-4 mb-6">
          <Select value={selectedTeam} onValueChange={setSelectedTeam}>
            <SelectTrigger className="w-full sm:w-[200px] bg-card border-border">
              <SelectValue placeholder="Filter by team" />
            </SelectTrigger>
            <SelectContent className="bg-popover border-border z-50">
              {laLigaTeams.map((team) => (
                <SelectItem key={team} value={team}>
                  {team}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={sortBy} onValueChange={setSortBy}>
            <SelectTrigger className="w-full sm:w-[200px] bg-card border-border">
              <SelectValue placeholder="Sort by" />
            </SelectTrigger>
            <SelectContent className="bg-popover border-border z-50">
              <SelectItem value="date">Date</SelectItem>
              <SelectItem value="confidence">Confidence</SelectItem>
              <SelectItem value="accuracy">Accuracy (History)</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Tabs */}
        <Tabs defaultValue="current" className="w-full">
          <TabsList className="w-full sm:w-auto bg-card border-border mb-6">
            <TabsTrigger
              value="current"
              className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              <Target className="w-4 h-4 mr-2" />
              Current Predictions
            </TabsTrigger>
            <TabsTrigger
              value="history"
              className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              <TrendingUp className="w-4 h-4 mr-2" />
              History
            </TabsTrigger>
          </TabsList>

          <TabsContent value="current" className="mt-0">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filterPredictions(currentPredictions).map((prediction) => (
                <MatchCard key={prediction.id} prediction={prediction} />
              ))}
            </div>
            {filterPredictions(currentPredictions).length === 0 && (
              <div className="text-center py-12 text-muted-foreground">
                No predictions found for the selected filters.
              </div>
            )}
          </TabsContent>

          <TabsContent value="history" className="mt-0">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filterPredictions(historicalPredictions).map((prediction) => (
                <MatchCard
                  key={prediction.id}
                  prediction={prediction}
                  showActual
                />
              ))}
            </div>
            {filterPredictions(historicalPredictions).length === 0 && (
              <div className="text-center py-12 text-muted-foreground">
                No historical predictions found for the selected filters.
              </div>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default Index;
