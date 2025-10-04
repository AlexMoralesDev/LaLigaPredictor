// src/lib/teamLogos.ts

// Using API-Football (logo.clearbit.com alternative) and fallbacks
export const teamLogos: Record<string, string> = {
  // La Liga team logos - using Cloudinary CDN (very reliable)
  "Deportivo Alavés": "https://media.api-sports.io/football/teams/720.png",
  "Deportivo AlavÃ©s": "https://media.api-sports.io/football/teams/720.png",

  "Athletic Club": "https://media.api-sports.io/football/teams/531.png",

  "RC Celta de Vigo": "https://media.api-sports.io/football/teams/538.png",
  "RC Celta": "https://media.api-sports.io/football/teams/538.png",

  "Girona FC": "https://media.api-sports.io/football/teams/547.png",
  Girona: "https://media.api-sports.io/football/teams/547.png",

  "Real Oviedo": "https://media.api-sports.io/football/teams/720.png",

  "Sevilla FC": "https://media.api-sports.io/football/teams/536.png",
  Sevilla: "https://media.api-sports.io/football/teams/536.png",

  "Real Sociedad de Fútbol":
    "https://media.api-sports.io/football/teams/548.png",
  "Real Sociedad de FÃºtbol":
    "https://media.api-sports.io/football/teams/548.png",
  "Real Sociedad": "https://media.api-sports.io/football/teams/548.png",

  "Real Madrid CF": "https://media.api-sports.io/football/teams/541.png",
  "Real Madrid": "https://media.api-sports.io/football/teams/541.png",

  "CA Osasuna": "https://media.api-sports.io/football/teams/727.png",
  Osasuna: "https://media.api-sports.io/football/teams/727.png",

  "RCD Espanyol de Barcelona":
    "https://media.api-sports.io/football/teams/540.png",
  Espanyol: "https://media.api-sports.io/football/teams/540.png",

  "Elche CF": "https://media.api-sports.io/football/teams/728.png",
  Elche: "https://media.api-sports.io/football/teams/728.png",

  "RCD Mallorca": "https://media.api-sports.io/football/teams/798.png",
  Mallorca: "https://media.api-sports.io/football/teams/798.png",

  "Club Atlético de Madrid":
    "https://media.api-sports.io/football/teams/530.png",
  "Club AtlÃ©tico de Madrid":
    "https://media.api-sports.io/football/teams/530.png",
  "Atletico Madrid": "https://media.api-sports.io/football/teams/530.png",
  "Atlético Madrid": "https://media.api-sports.io/football/teams/530.png",

  "Valencia CF": "https://media.api-sports.io/football/teams/532.png",
  Valencia: "https://media.api-sports.io/football/teams/532.png",

  "Levante UD": "https://cdn.sportmonks.com/images/soccer/teams/9/521.png",
  Levante: "https://cdn.sportmonks.com/images/soccer/teams/9/521.png",

  "FC Barcelona": "https://media.api-sports.io/football/teams/529.png",
  Barcelona: "https://media.api-sports.io/football/teams/529.png",

  "Rayo Vallecano de Madrid":
    "https://media.api-sports.io/football/teams/728.png",
  "Rayo Vallecano": "https://media.api-sports.io/football/teams/728.png",

  "Villarreal CF": "https://media.api-sports.io/football/teams/533.png",
  Villarreal: "https://media.api-sports.io/football/teams/533.png",

  "Getafe CF": "https://media.api-sports.io/football/teams/546.png",
  Getafe: "https://media.api-sports.io/football/teams/546.png",

  "Real Betis Balompié": "https://media.api-sports.io/football/teams/543.png",
  "Real Betis BalompiÃ©": "https://media.api-sports.io/football/teams/543.png",
  "Real Betis": "https://media.api-sports.io/football/teams/543.png",
  Betis: "https://media.api-sports.io/football/teams/543.png",
};

/**
 * Get team logo URL by team name with smart matching
 * @param teamName - The name of the team
 * @returns The logo URL or a fallback if not found
 */
export function getTeamLogo(teamName: string): string {
  console.log("🔍 Looking for logo for team:", teamName);

  // Direct match
  if (teamLogos[teamName]) {
    console.log("✅ Direct match found");
    return teamLogos[teamName];
  }

  // Try case-insensitive match
  const lowerTeamName = teamName.toLowerCase();
  for (const [key, value] of Object.entries(teamLogos)) {
    if (key.toLowerCase() === lowerTeamName) {
      console.log("✅ Case-insensitive match found:", key);
      return value;
    }
  }

  // Try partial match (contains)
  for (const [key, value] of Object.entries(teamLogos)) {
    const keyLower = key.toLowerCase();
    if (keyLower.includes(lowerTeamName) || lowerTeamName.includes(keyLower)) {
      console.log("✅ Partial match found:", key);
      return value;
    }
  }

  // Try matching without special characters
  const cleanTeamName = teamName
    .replace(/[^a-zA-Z0-9\s]/g, "")
    .toLowerCase()
    .trim();

  for (const [key, value] of Object.entries(teamLogos)) {
    const cleanKey = key
      .replace(/[^a-zA-Z0-9\s]/g, "")
      .toLowerCase()
      .trim();

    if (
      cleanKey === cleanTeamName ||
      cleanKey.includes(cleanTeamName) ||
      cleanTeamName.includes(cleanKey)
    ) {
      console.log("✅ Clean match found:", key);
      return value;
    }
  }

  console.warn("❌ No logo found for team:", teamName);
  console.warn(
    "Available teams:",
    Object.keys(teamLogos).slice(0, 5).join(", ") + "...",
  );

  // Generate fallback logo with team initials
  const initials = teamName
    .split(" ")
    .map((word) => word.charAt(0))
    .join("")
    .substring(0, 2)
    .toUpperCase();

  return `https://ui-avatars.com/api/?name=${encodeURIComponent(initials)}&background=1e293b&color=fff&size=128&bold=true&font-size=0.5`;
}
