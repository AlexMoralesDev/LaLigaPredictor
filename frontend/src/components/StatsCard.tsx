import { Card } from "@/components/ui/card";
import { TrendingUp } from "lucide-react";

interface StatsCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: "up" | "down" | "neutral";
}

const StatsCard = ({ title, value, subtitle, trend }: StatsCardProps) => {
  return (
    <Card className="p-6 bg-gradient-card border-border hover:border-accent/50 transition-all duration-300">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-muted-foreground mb-1">{title}</p>
          <p className="text-3xl font-bold text-foreground mb-1">{value}</p>
          {subtitle && (
            <p className="text-xs text-muted-foreground">{subtitle}</p>
          )}
        </div>
        {trend === "up" && (
          <div className="bg-success/20 p-2 rounded-lg">
            <TrendingUp className="w-5 h-5 text-success" />
          </div>
        )}
      </div>
    </Card>
  );
};

export default StatsCard;
