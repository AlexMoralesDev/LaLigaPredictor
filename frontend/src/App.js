import './App.css';
import GameWeekView from './components/GameWeekView';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        La Liga Predictor:
      </header>
      <GameWeekView matches={[{team1: 'Team A', team2: 'Team B'}, {team1: 'Team C', team2: 'Team D'}]} />
    </div>
  );
}

export default App;
