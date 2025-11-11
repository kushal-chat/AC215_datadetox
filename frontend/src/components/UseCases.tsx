import { Card, CardContent } from "@/components/ui/card";
import { AlertTriangle, CheckCircle2, FileWarning } from "lucide-react";

const useCases = [
  {
    icon: AlertTriangle,
    title: "LAION-5B Dataset Risks",
    problem: "Models trained on LAION-5B may contain copyrighted images and sensitive personal data",
    solution: "Identify all downstream models affected and assess legal compliance risks",
    color: "text-destructive",
  },
  {
    icon: FileWarning,
    title: "MS-Celeb-1M Withdrawal",
    problem: "Dataset withdrawn due to privacy concerns, but derivative models still in use",
    solution: "Track which models are impacted and provide migration recommendations",
    color: "text-accent",
  },
  {
    icon: CheckCircle2,
    title: "Compliance Verification",
    problem: "Need to verify model training data meets regulatory requirements",
    solution: "Automated lineage reports for audit and compliance documentation",
    color: "text-secondary",
  },
];

const UseCases = () => {
  return (
    <section className="py-24 px-6">
      <div className="container mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            Real-World Use Cases
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            See how DataDetox helps identify and mitigate risks in AI model deployment
          </p>
        </div>
        
        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {useCases.map((useCase, index) => (
            <Card 
              key={index} 
              className="bg-card border-border hover:shadow-lg transition-all duration-300 animate-fade-in"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <CardContent className="p-6">
                <div className={`w-12 h-12 mb-4 rounded-xl bg-gradient-to-br from-primary/10 to-secondary/10 flex items-center justify-center ${useCase.color}`}>
                  <useCase.icon className="w-6 h-6" />
                </div>
                <h3 className="text-xl font-bold mb-3">{useCase.title}</h3>
                <div className="space-y-3">
                  <div>
                    <div className="text-sm font-semibold text-destructive mb-1">Problem:</div>
                    <p className="text-sm text-muted-foreground">{useCase.problem}</p>
                  </div>
                  <div>
                    <div className="text-sm font-semibold text-secondary mb-1">Solution:</div>
                    <p className="text-sm text-muted-foreground">{useCase.solution}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default UseCases;
