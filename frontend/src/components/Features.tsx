import { Card, CardContent } from "@/components/ui/card";
import lineageIcon from "@/assets/feature-lineage.png";
import qaIcon from "@/assets/feature-qa.png";
import visualizationIcon from "@/assets/feature-visualization.png";
import riskIcon from "@/assets/feature-risk.png";

const features = [
  {
    icon: lineageIcon,
    title: "Model Lineage Tracking",
    description: "Trace complete dependency trees from upstream base models to downstream derivatives. Understand the full lineage of any AI model.",
  },
  {
    icon: qaIcon,
    title: "Natural Language Q&A",
    description: "Ask questions like 'Which models were trained on LAION-5B?' and get instant, structured answers powered by LLMs.",
  },
  {
    icon: visualizationIcon,
    title: "Interactive Visualization",
    description: "Explore model dependencies through clear, visual tree diagrams. Make complex relationships easy to understand.",
  },
  {
    icon: riskIcon,
    title: "Risk Detection",
    description: "Identify harmful datasets and potential compliance issues before they impact your project. Stay ahead of AI safety regulations.",
  },
];

const Features = () => {
  return (
    <section className="py-24 px-6 bg-muted/30">
      <div className="container mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            Powerful Features for
            <span className="bg-gradient-accent bg-clip-text text-transparent"> AI Safety</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Everything you need to understand and track AI model lineage in one interactive platform
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature, index) => (
            <Card
              key={index}
              className="bg-card border-border hover:shadow-lg transition-all duration-300 hover:-translate-y-1 animate-fade-in"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <CardContent className="p-6">
                <div className="w-16 h-16 mb-4 rounded-2xl bg-gradient-to-br from-primary/10 to-secondary/10 flex items-center justify-center">
                  <img src={feature.icon} alt={feature.title} className="w-10 h-10" />
                </div>
                <h3 className="text-xl font-bold mb-3">{feature.title}</h3>
                <p className="text-muted-foreground leading-relaxed">{feature.description}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
