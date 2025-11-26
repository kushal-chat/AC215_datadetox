import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import Features from "@/components/Features";
import UseCases from "@/components/UseCases";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <Hero />
      <Features />
      <UseCases />

      {/* Footer */}
      <footer className="py-12 px-6 border-t border-border bg-muted/30">
        <div className="container mx-auto text-center">
          <div className="mb-4">
            <span className="text-2xl font-bold bg-gradient-accent bg-clip-text text-transparent">
              DataDetox
            </span>
          </div>
          <p className="text-sm text-muted-foreground mb-2">
            Track AI Model Lineage, Prevent Hidden Risks
          </p>
          <p className="text-xs text-muted-foreground">
            AC 215 Milestone Project • Harvard University • 2025
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
