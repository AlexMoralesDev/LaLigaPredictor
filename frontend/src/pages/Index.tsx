import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import MatchCard from "@/components/MatchCard";
import StatsCard from "@/components/StatsCard";
import { currentPredictions, historicalPredictions, laLigaTeams } from "@/data/mockData";
import { Prediction } from "@/types/prediction";
import { Trophy, Target, TrendingUp } from "lucide-react";

const Index = () => {
  const [selectedTeam, setSelectedTeam] = useState("All Teams");
  const [sortBy, setSortBy] = useState("date");

  const filterPredictions = (predictions: Prediction[]) => {
    let filtered = [...predictions];

    if (selectedTeam !== "All Teams") {
      filtered = filtered.filter(
        (p) => p.homeTeam === selectedTeam || p.awayTeam === selectedTeam
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
    }

    return filtered;
  };

  const isCorrect = (prediction: Prediction) => {
    if (prediction.actualHomeScore === undefined) return false;
    const predictedResult = prediction.predictedHomeScore > prediction.predictedAwayScore ? "home" :
                           prediction.predictedHomeScore < prediction.predictedAwayScore ? "away" : "draw";
    const actualResult = prediction.actualHomeScore > prediction.actualAwayScore! ? "home" :
                        prediction.actualHomeScore < prediction.actualAwayScore! ? "away" : "draw";
    return predictedResult === actualResult;
  };

  const calculateStats = () => {
    const withActuals = historicalPredictions.filter(p => p.actualHomeScore !== undefined);
    const correct = withActuals.filter(isCorrect).length;
    const accuracy = withActuals.length > 0 ? ((correct / withActuals.length) * 100).toFixed(1) : "0";
    
    return {
      totalPredictions: withActuals.length,
      correctPredictions: correct,
      accuracy,
      avgConfidence: (historicalPredictions.reduce((sum, p) => sum + p.confidence, 0) / historicalPredictions.length).toFixed(1),
    };
  };

  const stats = calculateStats();

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="bg-gradient-primary border-b border-border">
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center gap-3 mb-2">
            <Trophy className="w-8 h-8 text-foreground" />
            <h1 className="text-4xl font-bold text-foreground">La Liga Predictor</h1>
          </div>
          <p className="text-foreground/80">AI-powered match predictions with historical accuracy tracking</p>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <StatsCard
            title="Overall Accuracy"
            value={`${stats.accuracy}%`}
            subtitle={`${stats.correctPredictions}/${stats.totalPredictions} correct`}
            trend="up"
          />
          <StatsCard
            title="Avg Confidence"
            value={`${stats.avgConfidence}%`}
            subtitle="Across all predictions"
          />
          <StatsCard
            title="Current Gameweek"
            value="8"
            subtitle="Active predictions"
          />
          <StatsCard
            title="Total Matches"
            value={currentPredictions.length + historicalPredictions.length}
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
              <SelectItem value="accuracy">Accuracy</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Tabs */}
        <Tabs defaultValue="current" className="w-full">
          <TabsList className="w-full sm:w-auto bg-card border-border mb-6">
            <TabsTrigger value="current" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              <Target className="w-4 h-4 mr-2" />
              Current Predictions
            </TabsTrigger>
            <TabsTrigger value="history" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
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
                <MatchCard key={prediction.id} prediction={prediction} showActual />
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
