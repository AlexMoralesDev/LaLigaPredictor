import { useState } from "react";
import {
  Brain,
  Code,
  Database,
  LineChart,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Linkedin,
} from "lucide-react";

const InfoFooter = () => {
  const [expandedSection, setExpandedSection] = useState<string | null>(null);

  const toggleSection = (section: string) => {
    setExpandedSection(expandedSection === section ? null : section);
  };

  const sections = [
    {
      id: "ml",
      icon: Brain,
      title: "Machine Learning Pipeline",
      content: {
        algorithm: "Random Forest Classifier with 300 decision trees",
        features:
          "50+ engineered features including performance metrics, form analysis, defensive/offensive statistics, venue effects, and head-to-head data",
        training:
          "2+ seasons of comprehensive La Liga match statistics with regular retraining cycles",
      },
    },
    {
      id: "tech",
      icon: Code,
      title: "Technology Stack",
      content: {
        frontend:
          "React 18 with TypeScript, Tailwind CSS, shadcn/ui components",
        backend: "Python 3.x, scikit-learn, pandas, NumPy",
        deployment: "Netlify with continuous deployment and automated testing",
      },
    },
    {
      id: "data",
      icon: Database,
      title: "Data Engineering",
      content: {
        pipeline: "Automated ETL pipeline for football statistics aggregation",
        sources: "Real-time data collection from football-data.org API",
        quality:
          "Data validation, quality checks, and error handling with logging",
      },
    },
    {
      id: "performance",
      icon: LineChart,
      title: "Performance Monitoring",
      content: {
        tracking: "Model accuracy tracking across different prediction types",
        metrics: "Calibration metrics to ensure probability accuracy",
        analysis:
          "Temporal analysis of prediction performance with A/B testing framework",
      },
    },
  ];

  return (
    <footer className="bg-gradient-primary border-t border-border mt-16">
      <div className="container mx-auto px-4 py-12">
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-foreground mb-2">
            About This Project
          </h2>
          <p className="text-foreground/80 max-w-3xl mx-auto">
            A full-stack machine learning application demonstrating end-to-end
            development from data collection and model training to deployment of
            a production-ready web interface
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
          {sections.map((section) => {
            const Icon = section.icon;
            const isExpanded = expandedSection === section.id;

            return (
              <div
                key={section.id}
                className="bg-card border border-border rounded-lg overflow-hidden transition-all duration-300 hover:border-primary/50"
              >
                <button
                  onClick={() => toggleSection(section.id)}
                  className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-secondary/50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <Icon className="w-5 h-5 text-primary" />
                    <span className="font-semibold text-foreground">
                      {section.title}
                    </span>
                  </div>
                  {isExpanded ? (
                    <ChevronUp className="w-5 h-5 text-muted-foreground" />
                  ) : (
                    <ChevronDown className="w-5 h-5 text-muted-foreground" />
                  )}
                </button>

                {isExpanded && (
                  <div className="px-6 pb-4 pt-2 space-y-3 animate-in fade-in slide-in-from-top-2 duration-300">
                    {Object.entries(section.content).map(([key, value]) => (
                      <div key={key}>
                        <span className="text-primary font-medium capitalize">
                          {key}:{" "}
                        </span>
                        <span className="text-foreground/80">{value}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-8 border-t border-border">
          <div className="text-center md:text-left">
            <h3 className="font-semibold text-foreground mb-2">
              Skills Demonstrated
            </h3>
            <p className="text-sm text-foreground/70">
              Machine Learning • Data Science • Full-Stack Development • DevOps
              • Software Engineering
            </p>
          </div>

          <div className="text-center">
            <h3 className="font-semibold text-foreground mb-2">
              Live Application
            </h3>
            <a
              href="https://laligapredictor.netlify.app"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-primary hover:text-primary/80 transition-colors inline-flex items-center gap-1"
            >
              laligapredictor.netlify.app
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>

          <div className="text-center md:text-right">
            <h3 className="font-semibold text-foreground mb-2">Developer</h3>
            <a
              href="https://www.linkedin.com/in/alex-morales-dev/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-primary hover:text-primary/80 transition-colors inline-flex items-center gap-1"
            >
              <Linkedin className="w-4 h-4" />
              Alex Morales Trevisan
            </a>
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
