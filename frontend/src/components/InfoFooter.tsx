import { useState, useEffect } from "react";
import {
  Brain,
  Code,
  Database,
  LineChart,
  ChevronDown,
  ChevronUp,
  Linkedin,
  Github,
  Youtube,
  Instagram,
  Sparkles,
  Zap,
  Target,
  TrendingUp,
  Users,
  Settings,
  Shield,
  Rocket,
} from "lucide-react";

const InfoFooter = () => {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(),
  );
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);
  const [animatedStats, setAnimatedStats] = useState({
    accuracy: 0,
    predictions: 0,
    components: 0,
  });

  // Animate stats on mount
  useEffect(() => {
    const duration = 2000;
    const steps = 60;
    const interval = duration / steps;

    const targets = {
      accuracy: 60,
      predictions: 300,
      components: 15,
    };

    let step = 0;
    const timer = setInterval(() => {
      step++;
      const progress = step / steps;
      const easeOut = 1 - Math.pow(1 - progress, 3);

      setAnimatedStats({
        accuracy: Math.round(targets.accuracy * easeOut),
        predictions: Math.round(targets.predictions * easeOut),
        components: Math.round(targets.components * easeOut),
      });

      if (step >= steps) {
        clearInterval(timer);
        setAnimatedStats(targets);
      }
    }, interval);

    return () => clearInterval(timer);
  }, []);

  const toggleSection = (section: string) => {
    setExpandedSections((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(section)) newSet.delete(section);
      else newSet.add(section);
      return newSet;
    });
  };

  const sections = [
    {
      id: "ml",
      icon: Brain,
      title: "Machine Learning Pipeline",
      gradient: "from-primary/20 to-accent/20",
      iconColor: "text-primary",
      content: {
        algorithm: "Random Forest Classifier with 300 decision trees",
        features:
          "50+ engineered features including performance metrics, form analysis, defensive/offensive statistics, venue effects, and head-to-head data",
        training:
          "2+ seasons of comprehensive La Liga match statistics with regular retraining cycles",
      },
      stats: { accuracy: "~60%", predictions: "300+", updates: "Weekly" },
    },
    {
      id: "tech",
      icon: Code,
      title: "Technology Stack",
      gradient: "from-accent/20 to-primary/20",
      iconColor: "text-accent",
      content: {
        frontend:
          "React 18 with TypeScript, Tailwind CSS, shadcn/ui components",
        backend: "Python 3.x, scikit-learn, pandas, NumPy",
        deployment: "Netlify with continuous deployment and automated testing",
      },
      stats: { components: "15+", typescript: "100%", responsive: "Yes" },
    },
    {
      id: "data",
      icon: Database,
      title: "Data Engineering",
      gradient: "from-success/20 to-success/10",
      iconColor: "text-success",
      content: {
        pipeline: "Automated ETL pipeline for football statistics aggregation",
        sources: "Real-time data collection from football-data.org API",
        quality:
          "Data validation, quality checks, and error handling with logging",
      },
      stats: { dataPoints: "10K+", sources: "3", automation: "Full" },
    },
    {
      id: "performance",
      icon: LineChart,
      title: "Performance Monitoring",
      gradient: "from-warning/20 to-warning/10",
      iconColor: "text-warning",
      content: {
        tracking: "Model accuracy tracking across different prediction types",
        metrics: "Calibration metrics to ensure probability accuracy",
        analysis:
          "Temporal analysis of prediction performance with A/B testing framework",
      },
      stats: { monitoring: "24/7", metrics: "20+", testing: "Continuous" },
    },
  ];

  const socialLinks = [
    {
      name: "LinkedIn",
      url: "https://www.linkedin.com/in/alex-morales-dev/",
      icon: Linkedin,
      color: "hover:text-[#0A66C2]",
    },
    {
      name: "GitHub",
      url: "https://github.com/AlexMoralesDev",
      icon: Github,
      color: "hover:text-foreground",
    },
    {
      name: "YouTube",
      url: "https://www.youtube.com/@alexmoralesdev",
      icon: Youtube,
      color: "hover:text-[#FF0000]",
    },
    {
      name: "Instagram",
      url: "https://www.instagram.com/alexmoralesdev?igsh=bzBrbnl0Nm0wOXJy&utm_source=qr",
      icon: Instagram,
      color: "hover:text-[#E4405F]",
    },
    {
      name: "TikTok",
      url: "https://www.tiktok.com/@alexmoralesdev",
      icon: () => (
        <svg viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
          <path d="M19.59 6.69a4.83 4.83 0 01-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 01-5.2 1.74 2.89 2.89 0 012.31-4.64 2.93 2.93 0 01.88.13V9.4a6.84 6.84 0 00-1-.05A6.33 6.33 0 005 20.1a6.34 6.34 0 0010.86-4.43v-7a8.16 8.16 0 004.77 1.52v-3.4a4.85 4.85 0 01-1-.1z" />
        </svg>
      ),
      color: "hover:text-[#00f2ea]",
    },
  ];

  const skills = [
    { label: "Machine Learning", icon: Brain },
    { label: "Data Science", icon: TrendingUp },
    { label: "Full-Stack Dev", icon: Code },
    { label: "DevOps", icon: Settings },
    { label: "Software Engineering", icon: Rocket },
  ];

  // Assign theme colors for alternating icons
  const skillColors = ["text-red-500", "text-success", "text-yellow-400"];
  const skillBgColors = [
    "bg-red-200/30",
    "bg-green-200/30",
    "bg-yellow-200/30",
  ];

  // Helper to get Tailwind class for a radial gradient based on text color
  const getRadialGradientClass = (textColorClass: string) => {
    switch (textColorClass) {
      case "text-success":
        return "from-success/30 to-transparent";
      case "text-accent":
        return "from-accent/30 to-transparent";
      case "text-warning":
        return "from-warning/30 to-transparent";
      case "text-primary":
        return "from-primary/30 to-transparent";
      case "text-red-500": // For skills, if needed (already handled there but good to generalize)
        return "from-red-500/30 to-transparent";
      case "text-yellow-400": // For skills
        return "from-yellow-400/30 to-transparent";
      default:
        return "from-gray-500/30 to-transparent";
    }
  };

  return (
    <footer className="bg-background border-t border-border mt-16">
      <div className="w-full px-4 py-12">
        {/* Footer Bottom */}
        <div className="w-full pt-8">
          <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8 px-4 md:px-8">
            {/* Project Highlights */}
            <div className="text-center md:text-left">
              <h3 className="font-semibold text-foreground mb-3 text-lg">
                Project Highlights
              </h3>
              <div className="space-y-2 text-sm text-muted-foreground">
                <div className="flex items-center gap-2 group cursor-default relative">
                  {" "}
                  {/* Added relative */}
                  <div
                    className={`absolute left-0 top-1/2 -translate-y-1/2 w-10 h-10 bg-gradient-to-radial ${getRadialGradientClass("text-success")} opacity-20 filter blur-2xl rounded-full`}
                  ></div>
                  <Shield className="w-4 h-4 text-success group-hover:scale-110 transition-transform relative z-10" />{" "}
                  {/* Added relative z-10 */}
                  <span className="relative z-10">
                    Production-ready deployment pipeline
                  </span>{" "}
                  {/* Added relative z-10 */}
                </div>
                <div className="flex items-center gap-2 group cursor-default relative">
                  {" "}
                  {/* Added relative */}
                  <div
                    className={`absolute left-0 top-1/2 -translate-y-1/2 w-10 h-10 bg-gradient-to-radial ${getRadialGradientClass("text-warning")} opacity-20 filter blur-2xl rounded-full`}
                  ></div>
                  <Zap className="w-4 h-4 text-warning group-hover:scale-110 transition-transform relative z-10" />
                  <span className="relative z-10">
                    Real-time data processing & predictions
                  </span>
                </div>
                <div className="flex items-center gap-2 group cursor-default relative">
                  {" "}
                  {/* Added relative */}
                  <div
                    className={`absolute left-0 top-1/2 -translate-y-1/2 w-10 h-10 bg-gradient-to-radial ${getRadialGradientClass("text-accent")} opacity-20 filter blur-2xl rounded-full`}
                  ></div>
                  <Users className="w-4 h-4 text-accent group-hover:scale-110 transition-transform relative z-10" />
                  <span className="relative z-10">
                    Responsive & accessible UI/UX design
                  </span>
                </div>
              </div>
            </div>

            {/* Connect With Me */}
            <div className="text-center md:text-right">
              <h3 className="font-semibold text-foreground mb-3 text-lg">
                Connect With Me
              </h3>
              <div className="flex flex-wrap justify-center md:justify-end gap-3 mb-3">
                {socialLinks.map((social) => {
                  const Icon = social.icon;
                  // No radial blur for social icons as per discussion, existing styles are good.
                  return (
                    <a
                      key={social.name}
                      href={social.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className={`p-2 bg-secondary rounded-lg border border-border transition-all duration-300 hover:scale-110 hover:border-primary/50 hover:-translate-y-1 ${social.color}`}
                      aria-label={social.name}
                    >
                      <Icon />
                    </a>
                  );
                })}
              </div>
              <p className="text-sm text-muted-foreground">
                Alex Morales Trevisan
              </p>
            </div>
          </div>
        </div>

        <div className="text-center mt-8 pt-6 border-t border-border">
          <p className="text-sm text-muted-foreground">
            © 2025 La Liga Predictor • Designed for analytical and educational
            purposes • Last Updated: October 2025
          </p>
        </div>
      </div>
    </footer>
  );
};

export default InfoFooter;
