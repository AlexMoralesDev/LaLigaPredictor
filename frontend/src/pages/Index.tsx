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
import { Trophy, Target, TrendingUp } from "lucide-react";

// Assuming these are your API functions and types
import { fetchAllPredictions, fetchModelStats } from "@/lib/api";
import { Prediction } from "@/types/prediction";

// Helper function (Consolidated and simplified)
const isCorrect = (prediction: Prediction): boolean => {
  // Only check for correctness if actual scores exist (match is complete)
  if (
    prediction.actualHomeScore === null ||
    prediction.actualAwayScore === null
  ) {
    return false;
  }

  // Use the Python-calculated 'isCorrect' boolean from the database if available
  if (typeof prediction.isCorrect === "boolean") {
    return prediction.isCorrect;
  }

  // Fallback: Compare predicted result (home/away/draw) to actual result
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

// Component Start
const Index = () => {
  const [selectedTeam, setSelectedTeam] = useState("All Teams");
  const [sortBy, setSortBy] = useState("date");
  const [allPredictions, setAllPredictions] = useState<Prediction[]>([]);
  const [modelStats, setModelStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);

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

  // 2. Separate current and historical predictions & generate team list
  const { currentPredictions, historicalPredictions, laLigaTeams } =
    useMemo(() => {
      const current: Prediction[] = [];
      const historical: Prediction[] = [];
      const teams = new Set<string>();

      const currentGW = modelStats?.currentGameweek;

      allPredictions.forEach((p) => {
        teams.add(p.homeTeam);
        teams.add(p.awayTeam);

        // Match is complete if actualHomeScore is not null.
        if (p.actualHomeScore !== null) {
          historical.push(p);
        } else if (p.gameweek === currentGW && p.actualHomeScore === null) {
          current.push(p);
        }
      });

      return {
        currentPredictions: current,
        historicalPredictions: historical.sort(
          (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime(),
        ),
        laLigaTeams: ["All Teams", ...Array.from(teams).sort()],
      };
    }, [allPredictions, modelStats]);

  // 3. Calculate Stats
  const calculateStats = useMemo(() => {
    const withActuals = historicalPredictions.filter(
      (p) => p.actualHomeScore !== null && p.actualHomeScore !== undefined,
    );
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
  }, [historicalPredictions, modelStats]);

  // 4. Filter and Sort Predictions
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
      filtered.sort((a, b) => {
        const aCorrect = isCorrect(a);
        const bCorrect = isCorrect(b);
        return (bCorrect ? 1 : 0) - (aCorrect ? 1 : 0);
      });
    } else {
      // Default sort by date ascending for current, or reverse for history
      filtered.sort(
        (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime(),
      );
    }
    return filtered;
  };

  // HEADER COMPONENT (Extracted for use in both loading and loaded states)
  const Header = (
    <header className="bg-gradient-primary border-b border-border">
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center gap-3 mb-2">
          <Trophy className="w-8 h-8 text-foreground" />
          <h1 className="text-4xl font-bold text-foreground">
            La Liga Predictor
          </h1>
        </div>
        <p className="text-foreground/80">
          ML learning powered match predictions with historical accuracy
          tracking
        </p>
      </div>
    </header>
  );

  // 5. Handle Loading State
  if (loading) {
    return (
      <div className="min-h-screen bg-background">
        {Header}
        <div className="flex items-center justify-center text-xl py-12">
          Loading Predictions...
        </div>
      </div>
    );
  }

  // 6. Main Render
  return (
    <div className="min-h-screen bg-background">
      {Header} {/* The Header is now here, guaranteed to show */}
      <div className="container mx-auto px-4 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <StatsCard
            title="Historical Accuracy"
            value={`${calculateStats.accuracy}%`}
            subtitle={`${calculateStats.correctPredictions}/${calculateStats.totalPredictions} correct`}
            trend="up"
          />
          <StatsCard
            title="Avg Confidence"
            value={`${calculateStats.avgConfidence}%`}
            subtitle={`Model Training: ${calculateStats.trainingAccuracy}%`}
          />
          <StatsCard
            title="Current Gameweek"
            value={calculateStats.currentGameweek}
            subtitle={`${currentPredictions.length} matches to be played`}
          />
          <StatsCard
            title="Total Matches"
            value={historicalPredictions.length + currentPredictions.length}
            subtitle="Predicted so far"
          />
        </div>

        {/* Filters and Sort */}
        <div className="flex flex-col sm:flex-row justify-between items-center mb-6 space-y-4 sm:space-y-0">
          <div className="flex space-x-4 w-full sm:w-auto">
            <Select value={selectedTeam} onValueChange={setSelectedTeam}>
              <SelectTrigger className="w-full sm:w-[200px] bg-card border-border">
                <SelectValue placeholder="Filter by Team" />
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
