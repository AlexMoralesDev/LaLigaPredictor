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

  return (
    <footer className="bg-background border-t border-border mt-16">
      <div className="w-full px-4 py-12">
        {/* Header */}
        <div className="w-full mb-12 text-center">
          <h2 className="text-4xl font-bold text-foreground mb-3 bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
            About This Project
          </h2>
          <p className="text-muted-foreground max-w-3xl mx-auto text-lg">
            A full-stack machine learning application demonstrating end-to-end
            development from data collection and model training to deployment of
            a production-ready web interface
          </p>
        </div>

        {/* Animated Stats */}
        <div className="w-full mb-8">
          <div className="max-w-7xl mx-auto bg-gradient-to-r from-primary/5 via-accent/5 to-warning/5 rounded-xl p-6 border border-border">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center group cursor-default">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <Target className="w-5 h-5 text-primary group-hover:scale-110 transition-transform" />
                  <div className="text-3xl font-bold text-primary">
                    {animatedStats.accuracy}%
                  </div>
                </div>
                <div className="text-sm text-muted-foreground">
                  Model Accuracy
                </div>
              </div>

              <div className="text-center group cursor-default">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <Zap className="w-5 h-5 text-accent group-hover:scale-110 transition-transform" />
                  <div className="text-3xl font-bold text-accent">
                    {animatedStats.predictions}+
                  </div>
                </div>
                <div className="text-sm text-muted-foreground">
                  Match Results Analyzed
                </div>
              </div>

              <div className="text-center group cursor-default">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <Sparkles className="w-5 h-5 text-warning group-hover:scale-110 transition-transform" />
                  <div className="text-3xl font-bold text-warning">Live</div>
                </div>
                <div className="text-sm text-muted-foreground">
                  Website Fully Deployed
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Sections */}
        <div className="w-full space-y-4 mb-12">
          <div className="max-w-7xl mx-auto space-y-4">
            {sections.map((section) => {
              const Icon = section.icon;
              const isExpanded = expandedSections.has(section.id);
              const isHovered = hoveredCard === section.id;

              return (
                <div
                  key={section.id}
                  onMouseEnter={() => setHoveredCard(section.id)}
                  onMouseLeave={() => setHoveredCard(null)}
                  className={`bg-card backdrop-blur-sm border border-border rounded-xl overflow-hidden transition-all duration-300 hover:shadow-xl hover:shadow-primary/10 ${
                    isHovered ? "border-primary/50" : ""
                  }`}
                >
                  <button
                    onClick={() => toggleSection(section.id)}
                    className="w-full px-6 py-5 flex items-center justify-between text-left hover:bg-secondary/50 transition-colors group"
                  >
                    <div className="flex items-center gap-4">
                      <div
                        className={`p-2 rounded-lg bg-gradient-to-br ${section.gradient} group-hover:scale-110 transition-transform duration-300`}
                      >
                        <Icon className={`w-6 h-6 ${section.iconColor}`} />
                      </div>
                      <span className="font-semibold text-foreground text-lg">
                        {section.title}
                      </span>
                    </div>
                    {isExpanded ? (
                      <ChevronUp className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors" />
                    ) : (
                      <ChevronDown className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors" />
                    )}
                  </button>

                  {isExpanded && (
                    <div className="px-6 pb-5 pt-2 space-y-4 animate-in fade-in slide-in-from-top-2 duration-300 border-t border-border/50">
                      <div className="flex flex-wrap gap-2 pt-2">
                        {Object.entries(section.stats).map(([key, value]) => (
                          <div
                            key={key}
                            className="px-3 py-1 bg-secondary/80 rounded-full text-xs font-medium text-muted-foreground border border-border hover:border-primary/50 hover:scale-105 transition-all cursor-default"
                          >
                            <span className="text-primary capitalize">
                              {key}:
                            </span>{" "}
                            {value}
                          </div>
                        ))}
                      </div>

                      {Object.entries(section.content).map(([key, value]) => (
                        <div
                          key={key}
                          className="pl-3 border-l-2 border-primary/30 hover:border-primary/60 transition-colors"
                        >
                          <span className="text-primary font-semibold capitalize block mb-1">
                            {key}
                          </span>
                          <span className="text-card-foreground text-sm leading-relaxed">
                            {value}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Skills */}
        <div className="w-full mb-8">
          <div className="max-w-7xl mx-auto bg-gradient-to-r from-primary/10 via-accent/10 to-primary/10 rounded-xl p-8 border border-border">
            <h3 className="font-bold text-foreground mb-4 text-xl text-center">
              Technical Skills Demonstrated
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {skills.map((skill) => {
                const SkillIcon = skill.icon;
                return (
                  <div
                    key={skill.label}
                    className="text-center p-4 bg-card rounded-lg hover:bg-secondary transition-all duration-300 hover:scale-105 cursor-default border border-border group"
                  >
                    <SkillIcon className="w-8 h-8 mx-auto mb-2 text-primary group-hover:text-accent transition-colors" />
                    <div className="text-sm font-medium text-foreground">
                      {skill.label}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Footer Bottom */}
        <div className="w-full pt-8 border-t border-border">
          <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8 px-4 md:px-8">
            {/* Project Highlights */}
            <div className="text-center md:text-left">
              <h3 className="font-semibold text-foreground mb-3 text-lg">
                Project Highlights
              </h3>
              <div className="space-y-2 text-sm text-muted-foreground">
                <div className="flex items-center gap-2 group cursor-default">
                  <Shield className="w-4 h-4 text-success group-hover:scale-110 transition-transform" />
                  <span>Production-ready deployment pipeline</span>
                </div>
                <div className="flex items-center gap-2 group cursor-default">
                  <Zap className="w-4 h-4 text-warning group-hover:scale-110 transition-transform" />
                  <span>Real-time data processing & predictions</span>
                </div>
                <div className="flex items-center gap-2 group cursor-default">
                  <Users className="w-4 h-4 text-accent group-hover:scale-110 transition-transform" />
                  <span>Responsive & accessible UI/UX design</span>
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
